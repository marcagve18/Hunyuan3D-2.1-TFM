"""skin_refine/refiner.py — SkinTextureRefiner Orchestrator
===========================================================
Multi-view face restoration pipeline using pluggable refiners.

Algorithm
---------
1. Forward-render the textured mesh from N viewpoints.
2. Apply a configurable face restorer to each rendered view.
3. Back-project the enhanced views onto the UV texture.
4. Blend: final_tex = blend_alpha × enhanced + (1-blend_alpha) × original

Usage
-----
    # Option 1: By name (auto-creates the refiner)
    refiner = SkinTextureRefiner(refiner_name='codeformer')
    
    # Option 2: Direct instance
    from skin_refine.refiners import CodeFormerRefiner
    face_restorer = CodeFormerRefiner()
    refiner = SkinTextureRefiner(refiner=face_restorer)
    
    # Option 3: Custom refiner
    class MyRefiner(BaseSkinRefiner):
        ...
    refiner = SkinTextureRefiner(refiner=MyRefiner())
    
    # Run
    refiner(render, debug_dir='debug')

Available refiners: 'gfpgan', 'codeformer', 'none'
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Optional

from DifferentiableRenderer.camera_utils import get_mv_matrix, transform_pos
from .base import BaseSkinRefiner
from .registry import create_refiner, list_refiners

logger = logging.getLogger(__name__)


def _save_img(arr, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def _render_view(render, elev: float, azim: float, resolution: int = 512,
                 tex_attr: str = "tex"):
    """Sample *tex_attr* texture from one viewpoint.

    Returns
    -------
    rendered : (H, W, 3) float tensor [0, 1]
    visible  : (H, W, 1) float mask
    """
    r_mv = get_mv_matrix(elev=elev, azim=azim,
                         camera_distance=render.camera_distance)
    pos_cam  = transform_pos(r_mv, render.vtx_pos, keepdim=True)
    pos_clip = transform_pos(render.camera_proj_mat, pos_cam)

    res = (resolution, resolution)
    rast_out, _ = render.raster_rasterize(pos_clip, render.pos_idx,
                                          resolution=res)

    uv, _ = render.raster_interpolate(render.vtx_uv[None, ...], rast_out,
                                      render.uv_idx)
    visible = (rast_out[0, ..., -1:] > 0).float()

    tex  = getattr(render, tex_attr)
    tex4 = tex.permute(2, 0, 1).unsqueeze(0)

    grid     = uv * 2.0 - 1.0
    rendered = F.grid_sample(tex4, grid, align_corners=True,
                             mode="bilinear", padding_mode="border")
    rendered = rendered.squeeze(0).permute(1, 2, 0) * visible
    return rendered, visible


class SkinTextureRefiner:
    """Multi-view face restoration orchestrator.

    Accepts a pluggable refiner (any BaseSkinRefiner subclass) or a
    refiner name string ('gfpgan', 'codeformer', 'none').

    Parameters
    ----------
    refiner : BaseSkinRefiner, optional
        Pre-constructed refiner instance. If None, use refiner_name.
    refiner_name : str, default='gfpgan'
        Name of refiner to auto-create. Ignored if refiner is provided.
    refiner_kwargs : dict, optional
        Keyword arguments passed to create_refiner().
    num_passes : int, default=2
        Iterations of render→restore→bake.
    blend_alpha : float, default=0.8
        UV blend (0=original, 1=fully enhanced).
    grain_strength : float, default=0.018
        Std-dev of luminance grain added to final UV.
    grain_seed : int, default=42
        RNG seed for reproducibility.
    refine_resolution : int, default=512
        Resolution used when rendering views.
    device : str, default='cuda'
        Device for operations.
    viewpoints : list of (elev, azim) tuples, optional
        Camera positions for multi-view rendering.
    """

    _DEFAULT_VIEWPOINTS = [
        (  0,   0),
        (  0,  30),
        (  0, -30),
        ( 15,   0),
        (-10,   0),
        (  0,  60),
        (  0, -60),
        ( 15,  30),
    ]

    def __init__(
        self,
        refiner: Optional[BaseSkinRefiner] = None,
        refiner_name: str = "gfpgan",
        refiner_kwargs: Optional[dict] = None,
        num_passes: int = 2,
        blend_alpha: float = 0.8,
        grain_strength: float = 0.018,
        grain_seed: int = 42,
        refine_resolution: int = 512,
        device: str = "cuda",
        viewpoints: Optional[list] = None,
    ):
        if refiner is None:
            logger.info(f"[SkinRefiner] Creating refiner: {refiner_name}")
            kwargs = refiner_kwargs or {}
            self.refiner = create_refiner(refiner_name, **kwargs)
        else:
            self.refiner = refiner

        self.num_passes = num_passes
        self.blend_alpha = blend_alpha
        self.grain_strength = grain_strength
        self.grain_seed = grain_seed
        self.refine_resolution = refine_resolution
        self.device = device
        self.viewpoints = viewpoints or self._DEFAULT_VIEWPOINTS

        logger.info(f"[SkinRefiner] Using refiner: {self.refiner.name}")

    def _restore_view(self, rendered_hwc: torch.Tensor) -> Optional[np.ndarray]:
        """Apply the plugged-in refiner to one rendered view."""
        pil_img = self.refiner._tensor_to_pil(rendered_hwc)
        restored = self.refiner.restore(pil_img)
        if restored is None:
            return None
        return self.refiner._pil_to_array(restored)

    def _run_one_pass(self, render, res, has_mr, debug_dir=None, pass_idx=0):
        """One render → restore → bake pass."""
        textures_enh, cos_maps_enh = [], []
        textures_mr,  cos_maps_mr  = [], []
        n_restored = 0

        for elev, azim in self.viewpoints:
            tag = f"az{azim:+04d}_el{elev:+03d}"
            rendered, _ = _render_view(render, elev, azim, res, "tex")

            if debug_dir and pass_idx == 0:
                _save_img(rendered.cpu().numpy(),
                          os.path.join(debug_dir, f"view_before_{tag}.png"))

            restored_rgb = self._restore_view(rendered)
            if restored_rgb is None:
                continue

            n_restored += 1
            enh_pil = Image.fromarray((restored_rgb * 255).clip(0, 255).astype(np.uint8))

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
                    f"restored {n_restored}/{len(self.viewpoints)} views.")

        if n_restored == 0:
            return None, None, None, None

        new_tex, valid_mask = render.fast_bake_texture(textures_enh, cos_maps_enh)
        new_tex_mr, valid_mask_mr = (
            render.fast_bake_texture(textures_mr, cos_maps_mr)
            if has_mr and textures_mr else (None, None)
        )
        return new_tex, valid_mask, new_tex_mr, valid_mask_mr

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

    def __call__(self, render, reference_images=None, debug_dir=None):
        """Enhance the texture currently loaded on *render* in-place.

        Parameters
        ----------
        render : MeshRender
            MeshRender with .tex (and optionally .tex_mr) set.
        reference_images : list of PIL Images, optional
            Reference images (passed to refiner; most refiners ignore this).
        debug_dir : str, optional
            Path to save before/after debug images.
        """
        res    = self.refine_resolution
        orig_tex    = render.tex.clone()
        has_mr      = hasattr(render, "tex_mr") and render.tex_mr is not None
        orig_tex_mr = render.tex_mr.clone() if has_mr else None

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            _save_img(orig_tex.cpu().numpy(),
                      os.path.join(debug_dir, "texture_input.png"))

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

        if self.grain_strength > 0:
            tex_final = render.tex.clone()
            gen = torch.Generator(device=tex_final.device)
            gen.manual_seed(self.grain_seed)
            noise = torch.randn(tex_final.shape, device=tex_final.device, generator=gen)
            luma_weights = torch.tensor([0.299, 0.587, 0.114],
                                        device=tex_final.device).view(1, 1, 3)
            luma = (tex_final * luma_weights).sum(-1, keepdim=True)
            noise_scaled = noise * (luma * 0.5 + 0.5) * self.grain_strength
            grained_tex = (tex_final + noise_scaled).clamp(0, 1)
            render.set_texture(grained_tex, force_set=True)
            logger.info(f"[SkinRefiner] Grain applied (σ={self.grain_strength}).")

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
