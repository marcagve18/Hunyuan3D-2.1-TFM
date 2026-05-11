"""skin_refine/refiners/flux_kontext.py — FLUX.1-Kontext + LoRA per-view refiner.
===================================================================================
Uses FluxKontextPipeline (black-forest-labs/FLUX.1-Kontext-dev) with a custom
LoRA to refine each rendered view before texture baking.

The pipeline conditions on the input image *and* a text prompt, making it ideal
for per-view refinement without losing the overall structure.

After Kontext inference a geometry-correction step is optionally applied:
  1. LoFTR (detector-free GPU matching) finds dense correspondences between
     the original render and the refined output.
  2. A homography (or affine fallback) maps the refined image back onto the
     input geometry.
  3. The result is masked to the input's face footprint so background
     predictions from Kontext don't leak into the texture atlas.
"""

import logging
from typing import Optional, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
_DEFAULT_LORA  = (
    "/home/maguilar/TFM/flux_kontext_lora/outputs/"
    "flux_kontext_lora_refiner_general/"
    "flux_kontext_lora_refiner_general.safetensors"
)


class FluxKontextRefiner(BaseSkinRefiner):
    """Per-view skin refiner backed by FLUX.1-Kontext-dev + a local LoRA.

    Parameters
    ----------
    lora_path : str
        Path to the local ``.safetensors`` LoRA file.
    lora_scale : float, default=1.2
        LoRA weight scale fused at load time.
    prompt : str
        Text prompt guiding the edit.
    guidance_scale : float, default=2.5
        Classifier-free guidance scale.
    num_inference_steps : int, default=30
        Denoising steps.
    width, height : int, default=1024
        Resolution at which FLUX runs (input is resized and output is
        resized back to the original size).
    seed : int, default=42
        RNG seed for reproducibility.
    model : str
        HuggingFace model id for the base Kontext pipeline.
    dtype : str, ``"bfloat16"`` or ``"float16"``
        Torch dtype for the pipeline weights.
    device : str, default=``"cuda"``
    align : bool, default=True
        If True, run LoFTR-based geometry correction after Kontext inference
        to warp the refined image back onto the input's geometry and mask it
        to the input's face footprint.
    align_conf : float, default=0.9
        LoFTR confidence threshold for keeping matches.
    align_feather : int, default=2
        Gaussian feather radius (px) for the face-mask edges.
    """

    def __init__(
        self,
        lora_path: str = _DEFAULT_LORA,
        lora_scale: float = 1.2,
        prompt: str = "make this person look real",
        guidance_scale: float = 2.5,
        num_inference_steps: int = 30,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        model: str = _DEFAULT_MODEL,
        dtype: str = "bfloat16",
        device: str = "cuda",
        align: bool = True,
        align_conf: float = 0.9,
        align_feather: int = 2,
    ):
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.seed = seed
        self.model = model
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self._device = device
        self.align = align
        self.align_conf = align_conf
        self.align_feather = align_feather
        self._pipeline = None
        self._loftr = None

    @property
    def name(self) -> str:
        return "FluxKontext-LoRA"

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from diffusers import FluxKontextPipeline

        logger.info(f"[FluxKontextRefiner] Loading pipeline: {self.model}")
        self._pipeline = FluxKontextPipeline.from_pretrained(
            self.model, torch_dtype=self.dtype
        ).to(self._device)

        logger.info(f"[FluxKontextRefiner] Loading LoRA: {self.lora_path}")
        self._pipeline.load_lora_weights(self.lora_path)
        self._pipeline.fuse_lora(lora_scale=self.lora_scale)
        logger.info("[FluxKontextRefiner] Ready.")

    def _load_loftr(self):
        if self._loftr is not None:
            return
        import kornia.feature as KF
        logger.info("[FluxKontextRefiner] Loading LoFTR …")
        self._loftr = KF.LoFTR(pretrained="indoor").to(self._device).eval()

    # ── geometry correction ───────────────────────────────────────────────────

    @staticmethod
    def _pil_to_gray_tensor(img: Image.Image, device: str) -> torch.Tensor:
        """PIL RGB → (1,1,H,W) float32 [0,1] grayscale."""
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return t.mean(dim=1, keepdim=True)

    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _correct_geometry(
        self, original: Image.Image, refined: Image.Image
    ) -> Image.Image:
        """Warp refined back onto original's geometry using LoFTR + homography."""
        self._load_loftr()

        inp_gray = self._pil_to_gray_tensor(original, self._device)
        ref_gray = self._pil_to_gray_tensor(refined,  self._device)

        with torch.no_grad():
            out = self._loftr({"image0": inp_gray, "image1": ref_gray})

        kpts0 = out["keypoints0"].cpu().numpy()
        kpts1 = out["keypoints1"].cpu().numpy()
        conf  = out["confidence"].cpu().numpy()

        good  = conf > self.align_conf
        kpts0, kpts1 = kpts0[good], kpts1[good]
        logger.info(f"[FluxKontextRefiner] LoFTR matches (conf>{self.align_conf}): {len(kpts0)}")

        if len(kpts0) > 0:
            disp = np.linalg.norm(kpts1 - kpts0, axis=1)
            logger.info(f"[FluxKontextRefiner] Displacement mean={disp.mean():.2f}px  "
                        f"max={disp.max():.2f}px")

        inp_bgr = self._pil_to_bgr(original)
        ref_bgr = self._pil_to_bgr(refined)
        H, W    = inp_bgr.shape[:2]

        # Compute transform (refined → input space)
        # Skip the warp if the transform is near-identity (avoids unnecessary
        # bilinear blur when Kontext preserves geometry well, which is typical).
        _IDENTITY_THRESHOLD = 2.0   # px — skip warp if mean displacement < this
        warped_bgr = ref_bgr
        if len(kpts0) >= 4:
            disp_mean = float(np.linalg.norm(kpts1 - kpts0, axis=1).mean()) if len(kpts0) else 0.0
            if disp_mean < _IDENTITY_THRESHOLD:
                logger.info(f"[FluxKontextRefiner] Mean displacement {disp_mean:.2f}px < "
                            f"{_IDENTITY_THRESHOLD}px threshold — skipping warp.")
            else:
                M, mask = cv2.findHomography(
                    kpts1.astype(np.float32), kpts0.astype(np.float32),
                    cv2.RANSAC, 3.0,
                )
                inliers = int(mask.sum()) if mask is not None else 0
                if M is not None and inliers >= 10:
                    logger.info(f"[FluxKontextRefiner] Homography inliers: {inliers}, "
                                f"mean disp: {disp_mean:.2f}px")
                    warped_bgr = cv2.warpPerspective(
                        ref_bgr, M, (W, H),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                else:
                    logger.info("[FluxKontextRefiner] Homography weak — trying affine …")
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
                        logger.warning("[FluxKontextRefiner] No valid transform — skipping warp.")
        else:
            logger.warning("[FluxKontextRefiner] Too few matches — skipping warp.")

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
        return self._bgr_to_pil(out_bgr)

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Refine a single rendered view with Flux Kontext + LoRA.

        Parameters
        ----------
        image : PIL Image (RGB)

        Returns
        -------
        refined : PIL Image (RGB) or None on failure
        """
        self._load_pipeline()

        try:
            original_size = image.size
            img_resized = image.convert("RGB").resize(
                (self.width, self.height), Image.LANCZOS
            )

            generator = torch.Generator(device=self._device).manual_seed(self.seed)

            logger.info(
                f"[FluxKontextRefiner] Refining {original_size} → "
                f"{self.width}×{self.height}, prompt={self.prompt!r}, "
                f"steps={self.num_inference_steps}, guidance={self.guidance_scale}"
            )

            result = self._pipeline(
                image=img_resized,
                prompt=self.prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                width=self.width,
                height=self.height,
                generator=generator,
            ).images[0]

            result = result.resize(original_size, Image.LANCZOS)

            if self.align:
                result = self._correct_geometry(image, result)

            return result

        except Exception as e:
            logger.warning(f"[FluxKontextRefiner] Refinement failed: {e}")
            return None

    def restore_batch(self, images: List[Image.Image]) -> List[Optional[Image.Image]]:
        """Refine a list of views sequentially (Flux is memory-heavy)."""
        self._load_pipeline()
        results = []
        for i, img in enumerate(images):
            logger.info(f"[FluxKontextRefiner] Processing view {i + 1}/{len(images)}…")
            results.append(self.restore(img))
        return results

    def to(self, device: str):
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self
