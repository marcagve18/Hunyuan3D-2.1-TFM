"""skin_refine/refiners/flux_klein_multiview.py — Klein 4B multi-view refiner.
=============================================================================
Same multi-view render→refine→bake loop as FluxKontextRefiner, but using
Klein 4B base (Flux2KleinPipeline) instead of Kontext.

Klein conditions on a reference image concatenated with noise latents,
so each rendered view is passed as ``image=`` to guide the generation.
This operates in screen-space via ``restore()`` — the orchestrator handles
the multi-view back-projection and baking.

Optionally applies LoFTR-based geometry correction (same as Kontext refiner).
"""

import logging
from typing import Optional, List

import cv2
import numpy as np
import torch
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


class FluxKleinMultiviewRefiner(BaseSkinRefiner):
    """Per-view skin refiner backed by Flux 2 Klein 4B.

    Parameters
    ----------
    model : str
        HuggingFace model ID for Klein 4B.
    lora_path : str, optional
        Path to a LoRA safetensors file to fuse into the model.
    lora_scale : float
        LoRA weight scale (only used if lora_path is set).
    prompt : str
        Text prompt guiding the edit.
    guidance_scale : float, default=2.5
        Classifier-free guidance scale.
    num_inference_steps : int, default=40
        Denoising steps.
    width, height : int, default=1024
        Resolution at which Klein runs (input is resized and output is
        resized back to the original size).
    seed : int, default=42
        RNG seed for reproducibility.
    dtype : str, ``"bfloat16"`` or ``"float16"``
    device : str, default=``"cuda"``
    align : bool, default=True
        If True, run LoFTR-based geometry correction after inference.
    align_conf : float, default=0.9
        LoFTR confidence threshold.
    align_feather : int, default=2
        Gaussian feather radius (px) for the face-mask edges.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        prompt: str = "Generate a high-resolution detailed version of image1, strictly following the edge structure of the image.",
        guidance_scale: float = 2.5,
        num_inference_steps: int = 40,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        dtype: str = "bfloat16",
        device: str = "cuda",
        align: bool = True,
        align_conf: float = 0.9,
        align_feather: int = 2,
        use_normals: bool = False,
    ):
        self.model_id = model
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.seed = seed
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self._device = device
        self.align = align
        self.align_conf = align_conf
        self.align_feather = align_feather
        self.use_normals = use_normals
        self._pipeline = None
        self._loftr = None
        self.debug_dir = None
        self._view_counter = 0
        self._current_normal = None

    @property
    def name(self) -> str:
        return "FluxKlein-Multiview"

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from diffusers import Flux2KleinPipeline
        import os

        logger.info(f"[FluxKleinMV] Loading pipeline: {self.model_id}")
        self._pipeline = Flux2KleinPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        )

        if self.lora_path and os.path.exists(self.lora_path):
            logger.info(f"[FluxKleinMV] Fusing LoRA: {self.lora_path}")
            self._pipeline.load_lora_weights(self.lora_path)
            self._pipeline.fuse_lora(lora_scale=self.lora_scale)

        self._pipeline.to(self._device)
        logger.info("[FluxKleinMV] Pipeline ready.")

    def _load_loftr(self):
        if self._loftr is not None:
            return
        import kornia.feature as KF
        logger.info("[FluxKleinMV] Loading LoFTR …")
        self._loftr = KF.LoFTR(pretrained="indoor").to(self._device).eval()

    # ------------------------------------------------------------------
    # Geometry correction (same approach as FluxKontextRefiner)
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_gray_tensor(img: Image.Image, device: str) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return t.mean(dim=1, keepdim=True)

    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _correct_geometry(
        self, original: Image.Image, refined: Image.Image, dbg_prefix: str = None
    ) -> Image.Image:
        self._load_loftr()

        inp_gray = self._pil_to_gray_tensor(original, self._device)
        ref_gray = self._pil_to_gray_tensor(refined, self._device)

        with torch.no_grad():
            out = self._loftr({"image0": inp_gray, "image1": ref_gray})

        kpts0 = out["keypoints0"].cpu().numpy()
        kpts1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

        good = conf > self.align_conf
        kpts0, kpts1 = kpts0[good], kpts1[good]
        logger.info(f"[FluxKleinMV] LoFTR matches (conf>{self.align_conf}): {len(kpts0)}")

        if dbg_prefix and len(kpts0) > 0:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(np.array(original)); ax[0].set_title("Original")
            ax[1].imshow(np.array(refined)); ax[1].set_title("Klein output")
            for p0, p1 in zip(kpts0, kpts1):
                ax[0].plot(p0[0], p0[1], 'r.', markersize=3)
                ax[1].plot(p1[0], p1[1], 'g.', markersize=3)
            for a in ax: a.axis("off")
            fig.savefig(f"{dbg_prefix}_keypoints.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
            ax2.imshow(np.array(original))
            for p0, p1 in zip(kpts0, kpts1):
                ax2.annotate("", xy=p1, xytext=p0,
                             arrowprops=dict(arrowstyle="->", color="red", lw=0.5))
            ax2.set_title(f"Displacement vectors ({len(kpts0)} matches)")
            ax2.axis("off")
            fig2.savefig(f"{dbg_prefix}_displacements.png", bbox_inches="tight", dpi=150)
            plt.close(fig2)

        inp_bgr = self._pil_to_bgr(original)
        ref_bgr = self._pil_to_bgr(refined)
        H, W = inp_bgr.shape[:2]

        _IDENTITY_THRESHOLD = 2.0
        warped_bgr = ref_bgr
        if len(kpts0) >= 4:
            disp_mean = float(np.linalg.norm(kpts1 - kpts0, axis=1).mean())
            if disp_mean < _IDENTITY_THRESHOLD:
                logger.info(f"[FluxKleinMV] Displacement {disp_mean:.2f}px < threshold — skipping warp.")
            else:
                M, mask = cv2.findHomography(
                    kpts1.astype(np.float32), kpts0.astype(np.float32),
                    cv2.RANSAC, 3.0,
                )
                inliers = int(mask.sum()) if mask is not None else 0
                if M is not None and inliers >= 10:
                    logger.info(f"[FluxKleinMV] Homography inliers: {inliers}, disp: {disp_mean:.2f}px")
                    warped_bgr = cv2.warpPerspective(
                        ref_bgr, M, (W, H),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                else:
                    A, mask = cv2.estimateAffine2D(
                        kpts1.astype(np.float32), kpts0.astype(np.float32),
                        method=cv2.RANSAC, ransacReprojThreshold=3.0,
                    )
                    if A is not None and int(mask.sum()) >= 4:
                        M3 = np.vstack([A, [0, 0, 1]])
                        warped_bgr = cv2.warpPerspective(
                            ref_bgr, M3, (W, H),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0),
                        )
                    else:
                        logger.warning("[FluxKleinMV] No valid transform — skipping warp.")
        else:
            logger.warning("[FluxKleinMV] Too few matches — skipping warp.")

        # Mask to input face footprint
        face_mask = (inp_bgr.max(axis=2) > 15).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        face_mask = cv2.dilate(face_mask, k, iterations=1)
        if self.align_feather > 0:
            weight = cv2.GaussianBlur(
                face_mask.astype(np.float32), (0, 0), sigmaX=self.align_feather
            ) / 255.0
        else:
            weight = face_mask.astype(np.float32) / 255.0

        out_bgr = (warped_bgr.astype(np.float32) * weight[:, :, None]).clip(0, 255).astype(np.uint8)

        if dbg_prefix:
            self._bgr_to_pil(warped_bgr).save(f"{dbg_prefix}_warped.png")
            Image.fromarray(face_mask).save(f"{dbg_prefix}_face_mask.png")
            self._bgr_to_pil(out_bgr).save(f"{dbg_prefix}_aligned_final.png")

        return self._bgr_to_pil(out_bgr)

    # ------------------------------------------------------------------
    # Per-view restore (dispatched by multi-view render→bake orchestrator)
    # ------------------------------------------------------------------

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        self._load_pipeline()

        try:
            original_size = image.size
            # Klein needs dimensions divisible by 16
            w16 = (self.width // 16) * 16
            h16 = (self.height // 16) * 16
            img_resized = image.convert("RGB").resize((w16, h16), Image.LANCZOS)

            generator = torch.Generator(device=self._device).manual_seed(self.seed)

            logger.info(
                f"[FluxKleinMV] Refining {original_size} → {w16}×{h16}, "
                f"prompt={self.prompt!r}, steps={self.num_inference_steps}, "
                f"guidance={self.guidance_scale}"
            )

            dbg_prefix = None
            if self.debug_dir:
                import os
                os.makedirs(self.debug_dir, exist_ok=True)
                dbg_prefix = os.path.join(self.debug_dir, f"view_{self._view_counter:02d}")
                image.save(f"{dbg_prefix}_input.png")

            klein_images = [img_resized]
            if self.use_normals and self._current_normal is not None:
                normal_resized = self._current_normal.convert("RGB").resize((w16, h16), Image.LANCZOS)
                klein_images.append(normal_resized)
                if dbg_prefix:
                    normal_resized.save(f"{dbg_prefix}_normal.png")
                logger.info("[FluxKleinMV] Passing normal map as additional reference image")

            result = self._pipeline(
                prompt=self.prompt,
                image=klein_images,
                height=h16,
                width=w16,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
            ).images[0]

            result = result.resize(original_size, Image.LANCZOS)

            if dbg_prefix:
                result.save(f"{dbg_prefix}_klein_output.png")

            if self.align:
                result = self._correct_geometry(image, result, dbg_prefix)

            if dbg_prefix:
                result.save(f"{dbg_prefix}_final.png")

            self._view_counter += 1
            return result

        except Exception as e:
            logger.warning(f"[FluxKleinMV] Refinement failed: {e}")
            return None

    def restore_batch(self, images: List[Image.Image]) -> List[Optional[Image.Image]]:
        self._load_pipeline()
        results = []
        for i, img in enumerate(images):
            logger.info(f"[FluxKleinMV] Processing view {i + 1}/{len(images)}…")
            results.append(self.restore(img))
        return results

    def to(self, device: str):
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self