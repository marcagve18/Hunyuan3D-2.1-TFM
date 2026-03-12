"""skin_refine/refiner.py — GFPGAN Face Restoration Texture Refiner
====================================================================
Post-processing module for Hunyuan3D-2.1 paint pipeline.

Algorithm
---------
1.  Forward-render the textured mesh from N viewpoints.
2.  Apply GFPGAN face restoration to each rendered view:
      – GFPGAN detects and aligns the face region.
      – Restores skin detail, colour, and texture at high quality.
      – Pastes the restored face back into the original render.
3.  Back-project the enhanced views onto the UV texture using the
    mesh's cosine-weighted baking (same as the original pipeline).
4.  Blend: final_tex = blend_alpha × enhanced + (1-blend_alpha) × original

Thesis contribution: applying pretrained face restoration (GFPGAN) in
screen space and mapping the result back to UV via differentiable
rendering, providing multi-view consistent skin enhancement.
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from DifferentiableRenderer.camera_utils import get_mv_matrix, transform_pos

logger = logging.getLogger(__name__)

# Default GFPGAN checkpoint URL (downloaded on first use)
_GFPGAN_CKPT_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.4/GFPGANv1.4.pth"
)


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_img(arr, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _pil_to_cv2(pil_img):
    """PIL RGB → cv2 BGR uint8."""
    import cv2
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(bgr):
    """cv2 BGR uint8 → PIL RGB."""
    import cv2
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable forward render
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _render_view(render, elev: float, azim: float, resolution: int = 512,
                 tex_attr: str = "tex"):
    """Sample *tex_attr* texture from one viewpoint.

    Returns
    -------
    rendered : (H, W, 3) float tensor [0, 1]
    visible  : (H, W, 1) float mask
    """
    device = render.device
    r_mv = get_mv_matrix(elev=elev, azim=azim,
                         camera_distance=render.camera_distance)
    pos_cam  = transform_pos(r_mv, render.vtx_pos, keepdim=True)
    pos_clip = transform_pos(render.camera_proj_mat, pos_cam)

    res = (resolution, resolution)
    rast_out, _ = render.raster_rasterize(pos_clip, render.pos_idx,
                                          resolution=res)

    uv, _ = render.raster_interpolate(render.vtx_uv[None, ...], rast_out,
                                      render.uv_idx)   # (1, H, W, 2) ∈ [0,1]
    visible = (rast_out[0, ..., -1:] > 0).float()     # (H, W, 1)

    tex  = getattr(render, tex_attr)                   # (H_t, W_t, 3)
    tex4 = tex.permute(2, 0, 1).unsqueeze(0)           # (1, 3, H_t, W_t)

    grid     = uv * 2.0 - 1.0
    rendered = F.grid_sample(tex4, grid, align_corners=True,
                             mode="bilinear", padding_mode="border")
    rendered = rendered.squeeze(0).permute(1, 2, 0) * visible  # (H, W, 3)
    return rendered, visible


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SkinTextureRefiner:
    """GFPGAN-based face restoration for baked UV textures.

    Enhances a baked UV texture by:
      1. Rendering the mesh from multiple viewpoints.
      2. Running GFPGAN face restoration on each rendered view.
      3. Baking the restored views back to UV using cosine-weighted blending.

    This is deterministic, stable, and produces photorealistic skin texture
    without adversarial training instability.
    """

    # (elev, azim) pairs — front-biased for face coverage
    _VIEWPOINTS = [
        (  0,   0),   # front
        (  0,  30),   # slight right
        (  0, -30),   # slight left
        ( 15,   0),   # slightly elevated front
        (-10,   0),   # slightly lowered front
        (  0,  60),   # right
        (  0, -60),   # left
        ( 15,  30),   # elevated right
    ]

    def __init__(
        self,
        ckpt_path: str      = None,
        upscale: int        = 2,        # process at 2× → finer GFPGAN detail
        arch: str           = "clean",
        channel_multiplier: int = 2,
        bg_upsampler         = None,
        restoration_strength: float = 0.6,
        num_passes: int      = 2,        # how many render→restore→bake iterations
        blend_alpha: float   = 0.8,
        grain_strength: float = 0.018,   # subtle luminance grain added to final UV
        grain_seed: int      = 42,
        refine_resolution: int = 512,
        device: str          = "cuda",
    ):
        """
        ckpt_path            path to GFPGANv1.4.pth; auto-downloaded if None
        upscale              internal GFPGAN upscale factor (2 = 1024px processing)
        restoration_strength GFPGAN weight [0,1]: 0=original, 1=fully restored
        num_passes           iterations of render→GFPGAN→bake (2 = refine twice)
        blend_alpha          final UV blend (0=original, 1=fully enhanced)
        grain_strength       std-dev of luminance grain added after enhancement
        refine_resolution    resolution used when rendering views
        """
        self.restoration_strength = restoration_strength
        self.num_passes            = num_passes
        self.blend_alpha           = blend_alpha
        self.grain_strength        = grain_strength
        self.grain_seed            = grain_seed
        self.refine_resolution     = refine_resolution
        self.device                = device

        # Resolve checkpoint path
        if ckpt_path is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            ckpt_path = os.path.join(_here, "..", "..", "ckpt", "GFPGANv1.4.pth")

        if not os.path.isfile(ckpt_path):
            logger.info(f"[SkinRefiner] Downloading GFPGAN checkpoint → {ckpt_path}")
            import urllib.request
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            urllib.request.urlretrieve(_GFPGAN_CKPT_URL, ckpt_path)

        logger.info(f"[SkinRefiner] Loading GFPGAN from {ckpt_path} …")
        from gfpgan import GFPGANer
        self.restorer = GFPGANer(
            model_path=ckpt_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler,
        )
        logger.info("[SkinRefiner] GFPGAN ready.")

    # ------------------------------------------------------------------
    def _restore_view(self, rendered_hwc: torch.Tensor) -> np.ndarray:
        """Apply GFPGAN to one rendered view.

        Parameters
        ----------
        rendered_hwc : (H, W, 3) float tensor [0, 1]

        Returns
        -------
        restored_rgb : (H, W, 3) uint8 numpy array, or None if no face found
        """
        import cv2
        np_rgb   = (rendered_hwc.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr_in   = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        _, _, bgr_out = self.restorer.enhance(
            bgr_in,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=self.restoration_strength,
        )

        if bgr_out is None:
            return None
        return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    def _run_one_pass(self, render, res, has_mr, debug_dir=None, pass_idx=0):
        """One render → GFPGAN → bake pass. Returns (new_tex, valid_mask, new_tex_mr, valid_mask_mr)."""
        textures_enh, cos_maps_enh = [], []
        textures_mr,  cos_maps_mr  = [], []
        n_restored = 0

        for elev, azim in self._VIEWPOINTS:
            tag = f"az{azim:+04d}_el{elev:+03d}"
            rendered, _ = _render_view(render, elev, azim, res, "tex")

            if debug_dir and pass_idx == 0:
                _save_img(rendered.cpu().numpy(),
                          os.path.join(debug_dir, f"view_before_{tag}.png"))

            restored_rgb = self._restore_view(rendered)
            if restored_rgb is None:
                continue

            n_restored += 1
            enh_pil = Image.fromarray(restored_rgb)

            if debug_dir and pass_idx == self.num_passes - 1:
                enh_pil.save(os.path.join(debug_dir, f"view_after_{tag}.png"))

            tex_bp, cos_bp, _ = render.back_project(enh_pil, elev, azim)
            textures_enh.append(tex_bp)
            cos_maps_enh.append(cos_bp)

            if has_mr:
                mr_rendered, _ = _render_view(render, elev, azim, res, "tex_mr")
                mr_pil = Image.fromarray(
                    (mr_rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
                tex_mr_bp, cos_mr_bp, _ = render.back_project(mr_pil, elev, azim)
                textures_mr.append(tex_mr_bp)
                cos_maps_mr.append(cos_mr_bp)

        logger.info(f"[SkinRefiner] Pass {pass_idx+1}/{self.num_passes}: "
                    f"restored {n_restored}/{len(self._VIEWPOINTS)} views.")

        if n_restored == 0:
            return None, None, None, None

        new_tex, valid_mask = render.fast_bake_texture(textures_enh, cos_maps_enh)
        new_tex_mr, valid_mask_mr = (
            render.fast_bake_texture(textures_mr, cos_maps_mr)
            if has_mr and textures_mr else (None, None)
        )
        return new_tex, valid_mask, new_tex_mr, valid_mask_mr

    # ------------------------------------------------------------------
    def _to_float_tex(self, t, ref_device):
        if t is None:
            return None
        if t.max() > 1.5:
            t = t.float() / 255.0
        return torch.as_tensor(t, dtype=torch.float32, device=ref_device)

    def _blend(self, new_tex, valid_mask, orig_tex):
        α = self.blend_alpha
        mask = torch.as_tensor(valid_mask, dtype=torch.float32, device=orig_tex.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        return (α * new_tex * mask + orig_tex * (1 - α * mask)).clamp(0, 1)

    # ------------------------------------------------------------------
    def __call__(self, render, reference_images=None, debug_dir=None):
        """Enhance the texture currently loaded on *render* in-place.

        Parameters
        ----------
        render            : MeshRender with .tex (and optionally .tex_mr) set
        reference_images  : unused (kept for API compatibility)
        debug_dir         : optional path; saves before/after renders if given
        """
        res    = self.refine_resolution
        device = self.device

        # ── 0. Save original ─────────────────────────────────────────────
        orig_tex    = render.tex.clone()
        has_mr      = hasattr(render, "tex_mr") and render.tex_mr is not None
        orig_tex_mr = render.tex_mr.clone() if has_mr else None

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            _save_img(orig_tex.cpu().numpy(),
                      os.path.join(debug_dir, "texture_input.png"))

        # ── 1. Multi-pass GFPGAN enhancement ─────────────────────────────
        for pass_idx in range(self.num_passes):
            new_tex, valid_mask, new_tex_mr, valid_mask_mr = self._run_one_pass(
                render, res, has_mr, debug_dir, pass_idx)

            if new_tex is None:
                logger.warning("[SkinRefiner] No views restored — stopping.")
                break

            new_tex = self._to_float_tex(new_tex, orig_tex.device)
            final_tex = self._blend(new_tex, valid_mask, render.tex)

            if has_mr and new_tex_mr is not None:
                new_tex_mr = self._to_float_tex(new_tex_mr, orig_tex_mr.device)
                final_tex_mr = self._blend(new_tex_mr, valid_mask_mr, render.tex_mr)
                render.set_texture_mr(final_tex_mr, force_set=True)

            render.set_texture(final_tex, force_set=True)

        # ── 2. Add skin grain to final UV ─────────────────────────────────
        if self.grain_strength > 0:
            tex_final = render.tex.clone()
            gen = torch.Generator(device=tex_final.device)
            gen.manual_seed(self.grain_seed)
            # Luminance-weighted noise: looks like fine skin texture variation
            noise = torch.randn(tex_final.shape, device=tex_final.device, generator=gen)
            # Convert to luminance to apply grain only in luma channel
            luma_weights = torch.tensor([0.299, 0.587, 0.114],
                                        device=tex_final.device).view(1, 1, 3)
            luma = (tex_final * luma_weights).sum(-1, keepdim=True)   # (H, W, 1)
            # Scale noise by local luminance so highlights/shadows aren't over-grained
            noise_scaled = noise * (luma * 0.5 + 0.5) * self.grain_strength
            grained_tex = (tex_final + noise_scaled).clamp(0, 1)
            render.set_texture(grained_tex, force_set=True)
            logger.info(f"[SkinRefiner] Grain applied (σ={self.grain_strength}).")

        # ── 3. Debug saves ────────────────────────────────────────────────
        if debug_dir:
            final_tex = render.tex
            _save_img(final_tex.cpu().numpy(),
                      os.path.join(debug_dir, "texture_output.png"))
            before_np = np.clip(orig_tex.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            after_np  = np.clip(final_tex.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            sep = np.ones((before_np.shape[0], 4, 3), dtype=np.uint8) * 200
            Image.fromarray(np.concatenate([before_np, sep, after_np], axis=1)).save(
                os.path.join(debug_dir, "texture_comparison.png"))
            logger.info(f"[SkinRefiner] Debug images → {debug_dir}")

        logger.info("[SkinRefiner] Enhancement complete.")
