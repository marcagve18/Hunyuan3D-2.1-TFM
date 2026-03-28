"""skin_refine/refiners/screen_freq.py — Screen-space Laplacian back-projection refiner.
==========================================================================================
Fixes the core limitation of FreqSkin (UV-space SD): SD doesn't recognise UV-mapped
textures as faces, so it produces no useful high-frequency detail.

Algorithm
---------
For each viewpoint:
  1. Render the current UV texture to screen space  →  orig_screen
  2. Run SD img2img on orig_screen (SD understands rendered faces)  →  sd_screen
  3. Laplacian-decompose both views
  4. Compute per-band signed residual:
       hf_residual = Σ_i  w_i * (bands_sd[i] − bands_orig[i])   for i < freq_split
  5. Shift residual by +0.5 so it lives in [0,1]  →  hf_shifted
  6. Back-project hf_shifted to UV space via the differentiable renderer
  7. Unshift back to signed residual in UV space

Accumulate cosine-weighted residuals across all views, then add to the original UV.

Why this works
--------------
- SD operates on rendered face views where it was trained → produces real pore detail
- Only high-freq Laplacian residuals are back-projected → no color/shading seams
- Low-frequency UV texture is always preserved from the original bake
- View seams don't accumulate because residuals are near-zero at silhouette edges

Debug outputs (written to debug_dir when provided)
--------------------------------------------------
  view_orig_{tag}.png      — rendered view before SD
  view_sd_{tag}.png        — SD output for that view
  view_hf_{tag}.png        — high-freq residual (shifted) for that view
  hf_residual_uv.png       — accumulated UV residual (shifted, for visualisation)
  screen_freq_output.png   — final refined UV texture
  comparison.png           — [original UV | residual map | final UV] side by side
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from PIL import Image

from ..base import BaseSkinRefiner
from .freqskin import build_laplacian_pyramid, _upsample

logger = logging.getLogger(__name__)

_SD15_PRETRAINED = "SG161222/Realistic_Vision_V5.1_noVAE"
_SD15_VAE        = "stabilityai/sd-vae-ft-mse"   # recommended by model author (diffusers format)
_SD15_CONTROLNET = "lllyasviel/control_v11f1e_sd15_tile"

_NEGATIVE_PROMPT = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
    "text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, "
    "morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, "
    "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, "
    "gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, "
    "fused fingers, too many fingers, long neck"
)


def _save_png(arr_or_tensor, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(arr_or_tensor, torch.Tensor):
        arr_or_tensor = arr_or_tensor.detach().cpu().numpy()
    arr = np.clip(arr_or_tensor * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class ScreenFreqSkinRefiner(BaseSkinRefiner):
    """Screen-space Laplacian high-frequency back-projection refiner.

    Parameters
    ----------
    num_levels : int
        Laplacian pyramid levels (default: 4).
    freq_split : int
        Number of high-freq bands to take from SD (default: 2).
        Bands 0..(freq_split-1) are injected; rest are from original.
    render_resolution : int
        Resolution for screen-space rendering and SD processing (default: 512).
    strength : float
        SD img2img strength (default: 0.35). Higher is fine here because we
        only back-project the high-freq residual, not the full colour image.
    controlnet_scale : float
        ControlNet conditioning scale (default: 1.0).
    guidance_scale : float
        CFG scale (default: 5.0).
    num_inference_steps : int
        Diffusion steps (default: 25).
    prompt : str
        Positive prompt. Should describe the face in screen space.
    negative_prompt : str, optional
        Negative prompt.
    band_weights : list of float
        Per-band blend weight for injected high-freq bands.
    seed : int
        RNG seed (default: 42).
    pretrained_path : str
        SD 1.5 checkpoint.
    vae_path : str
        VAE checkpoint.
    controlnet_path : str
        ControlNet Tile checkpoint.
    device : str
        Device (default: 'cuda').
    viewpoints : list of (elev, azim) tuples, optional
        Camera positions for rendering. Defaults to face-focused set.
    """

    _DEFAULT_VIEWPOINTS = [
        (  0,   0),   # front
        (  0,  30),   # slight right
        (  0, -30),   # slight left
        (  0,  60),   # right
        (  0, -60),   # left
        ( 30,   0),   # top-front
        ( 30,  45),   # top-right
        ( 30, -45),   # top-left
        (  0, 180),   # back
    ]

    def __init__(
        self,
        num_levels: int = 4,
        freq_split: int = 2,
        render_resolution: int = 896,
        strength: float = 0.25,
        controlnet_scale: float = 1.0,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 25,
        prompt: str = "photorealistic baby face, natural skin pores, subsurface scattering, detailed skin texture",
        negative_prompt: Optional[str] = None,
        band_weights: Optional[List[float]] = None,
        seed: int = 42,
        pretrained_path: str = _SD15_PRETRAINED,
        vae_path: str = _SD15_VAE,
        controlnet_path: str = _SD15_CONTROLNET,
        device: str = "cuda",
        viewpoints: Optional[List[Tuple[float, float]]] = None,
    ):
        self.num_levels = num_levels
        self.freq_split = freq_split
        self.render_resolution = render_resolution
        self.strength = strength
        self.controlnet_scale = controlnet_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.prompt = prompt
        self.negative_prompt = negative_prompt or _NEGATIVE_PROMPT
        self.band_weights = band_weights or [0.8] * freq_split
        self.seed = seed
        self.pretrained_path = pretrained_path
        self.vae_path = vae_path
        self.controlnet_path = controlnet_path
        self._device = device
        self.viewpoints = viewpoints or self._DEFAULT_VIEWPOINTS
        self._pipeline = None

    @property
    def name(self) -> str:
        return "ScreenFreq-SD15"

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from diffusers import ControlNetModel, AutoencoderKL
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        from diffusers import DPMSolverMultistepScheduler

        logger.info("[ScreenFreq] Loading ControlNet Tile…")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path, torch_dtype=torch.float16
        )
        logger.info("[ScreenFreq] Loading VAE…")
        vae = AutoencoderKL.from_pretrained(self.vae_path, torch_dtype=torch.float16)
        logger.info(f"[ScreenFreq] Loading SD 1.5 ({self.pretrained_path})…")
        self._pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.pretrained_path,
            vae=vae,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        self._pipeline.to(self._device)
        self._pipeline.safety_checker = None
        # DPM++ 2M Karras — recommended family for Realistic Vision V6, no extra deps
        self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            final_sigmas_type="sigma_min",
        )
        self._pipeline.set_progress_bar_config(disable=True)
        logger.info("[ScreenFreq] SD 1.5 pipeline ready (DPM++ 2M Karras).")

    def _unload_pipeline(self):
        if self._pipeline is not None:
            self._pipeline.to("cpu")
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SD inference
    # ------------------------------------------------------------------

    def _run_sd(self, image: Image.Image) -> Optional[Image.Image]:
        """Run SD ControlNet Tile img2img on a single screen-space view."""
        w, h = image.size
        w8, h8 = (w // 8) * 8, (h // 8) * 8
        if (w8, h8) != (w, h):
            image = image.resize((w8, h8), Image.LANCZOS)

        generator = torch.Generator(device=self._device).manual_seed(self.seed)
        try:
            result = self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=image,
                control_image=image,
                strength=self.strength,
                controlnet_conditioning_scale=self.controlnet_scale,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                guess_mode=False,
            ).images[0]
            if result.size != (w, h):
                result = result.resize((w, h), Image.LANCZOS)
            return result
        except Exception as e:
            logger.warning(f"[ScreenFreq] SD failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Core: screen-space Laplacian back-projection
    # ------------------------------------------------------------------

    def refine_screen_freq(self, render, debug_dir: Optional[str] = None):
        """Refine UV texture via screen-space Laplacian back-projection.

        Called by the SkinTextureRefiner orchestrator when this refiner is active.
        Modifies render.tex in-place.

        Parameters
        ----------
        render : MeshRender
        debug_dir : str, optional
        """
        from ..refiner import _render_view

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        orig_tex = render.tex.clone()
        dev = orig_tex.device
        res = self.render_resolution

        # Accumulators in UV space
        residual_accum = torch.zeros_like(orig_tex)          # signed residual sum
        weight_accum   = torch.zeros(orig_tex.shape[0], orig_tex.shape[1], 1, device=dev)

        self._load_pipeline()
        n_processed = 0

        for elev, azim in self.viewpoints:
            tag = f"az{azim:+04d}_el{elev:+03d}"

            # 1. Render current texture to screen space
            rendered, visible = _render_view(render, elev, azim, res, "tex")
            if visible.sum() < 500:
                logger.debug(f"[ScreenFreq] View {tag}: too few visible pixels, skipping.")
                continue

            # 2. Convert to PIL
            orig_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_np)

            if debug_dir:
                orig_pil.save(os.path.join(debug_dir, f"view_orig_{tag}.png"))

            # 3. Run SD
            sd_pil = self._run_sd(orig_pil)
            if sd_pil is None:
                continue

            if debug_dir:
                sd_pil.save(os.path.join(debug_dir, f"view_sd_{tag}.png"))

            # 4. To float tensors on device
            orig_screen = torch.from_numpy(
                np.array(orig_pil).astype(np.float32) / 255.0
            ).to(dev)
            sd_screen = torch.from_numpy(
                np.array(sd_pil).astype(np.float32) / 255.0
            ).to(dev)

            # 5. Laplacian decompose both
            bands_orig, _ = build_laplacian_pyramid(orig_screen, self.num_levels)
            bands_sd, _   = build_laplacian_pyramid(sd_screen,   self.num_levels)

            # 6. Compute high-freq residual (sum of weighted band differences)
            #    Each band_diff[i] lives at resolution res / 2^i;
            #    upsample all to full render resolution before summing.
            hf_residual = torch.zeros_like(orig_screen)
            for i in range(self.freq_split):
                w = self.band_weights[i] if i < len(self.band_weights) else 0.8
                diff = bands_sd[i] - bands_orig[i]
                if diff.shape[:2] != orig_screen.shape[:2]:
                    diff = _upsample(diff, orig_screen.shape[0], orig_screen.shape[1])
                hf_residual = hf_residual + w * diff

            # 7. Shift to [0,1] for back_project (residual ≈ [-0.5, 0.5])
            hf_shifted = (hf_residual + 0.5).clamp(0, 1)

            if debug_dir:
                _save_png(hf_shifted, os.path.join(debug_dir, f"view_hf_{tag}.png"))

            # 8. Back-project shifted residual to UV
            hf_pil = Image.fromarray(
                (hf_shifted.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            )
            tex_bp, cos_bp, _ = render.back_project(hf_pil, elev, azim)

            # Convert back-projected values to float [0,1]
            tex_bp_t = torch.as_tensor(tex_bp, dtype=torch.float32, device=dev)
            if tex_bp_t.max() > 1.5:
                tex_bp_t = tex_bp_t / 255.0

            # Unshift: UV residual is now signed
            uv_residual = tex_bp_t - 0.5

            # Cosine weight
            cos_t = torch.as_tensor(cos_bp, dtype=torch.float32, device=dev)
            if cos_t.ndim == 2:
                cos_t = cos_t.unsqueeze(-1)

            residual_accum += uv_residual * cos_t
            weight_accum   += cos_t
            n_processed += 1
            logger.info(f"[ScreenFreq] View {tag}: processed.")

        self._unload_pipeline()

        if n_processed == 0:
            logger.warning("[ScreenFreq] No views processed — texture unchanged.")
            return

        # 9. Normalise and inject residual into original UV
        avg_residual = residual_accum / weight_accum.clamp(min=1e-8)
        refined = (orig_tex + avg_residual).clamp(0, 1)

        if debug_dir:
            # Visualise the residual map (shifted back to [0,1] for display)
            _save_png((avg_residual + 0.5).clamp(0, 1),
                      os.path.join(debug_dir, "hf_residual_uv.png"))
            _save_png(refined, os.path.join(debug_dir, "screen_freq_output.png"))

            # Comparison: [original | residual map | final]
            sep = np.ones((orig_tex.shape[0], 6, 3), dtype=np.uint8) * 180
            o = (orig_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            r = ((avg_residual + 0.5).clamp(0, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            f = (refined.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(np.concatenate([o, sep, r, sep, f], axis=1)).save(
                os.path.join(debug_dir, "comparison.png")
            )
            logger.info(f"[ScreenFreq] Debug images saved to {debug_dir}")

        render.set_texture(refined, force_set=True)
        logger.info(f"[ScreenFreq] Done. Processed {n_processed}/{len(self.viewpoints)} views.")

    # ------------------------------------------------------------------
    # BaseSkinRefiner fallback (required by ABC)
    # ------------------------------------------------------------------

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Fallback screen-space restore (used if called outside orchestrator)."""
        self._load_pipeline()
        try:
            return self._run_sd(image)
        except Exception as e:
            logger.warning(f"[ScreenFreq] restore() failed: {e}")
            return None

    def to(self, device: str):
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self
