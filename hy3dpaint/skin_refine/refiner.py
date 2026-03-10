"""
SkinTextureRefiner
==================
Post-processing module for the Hunyuan3D-2.1 paint pipeline.

Optimises the baked UV texture by maximising the score of a frozen
pretrained StyleGAN2-FFHQ discriminator on rendered views.

The discriminator is kept completely frozen — only the texture pixels
are updated.  Identity + TV regularisation prevent the texture from
drifting far from the inpainted original.

Usage (inserted in textureGenPipeline.py after inpainting):

    texture, texture_mr = self.models["skin_refiner"](
        texture=texture,
        texture_mr=texture_mr,
        render=self.render,
        reference_images=image_style,
        debug_dir=debug_dir,
    )
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .discriminator import StyleGAN2Discriminator
from .losses import adversarial_loss_G, total_variation

try:
    from DifferentiableRenderer.camera_utils import get_mv_matrix, transform_pos
except ImportError:
    from camera_utils import get_mv_matrix, transform_pos


# ---------------------------------------------------------------------------
# UV-map extraction helpers
# ---------------------------------------------------------------------------

def _extract_uv_maps(render, viewpoints, resolution: int = 512):
    device = render.device
    proj   = render.camera_proj_mat
    uv_maps: Dict = {}
    masks:   Dict = {}

    with torch.no_grad():
        for azim, elev in viewpoints:
            r_mv = get_mv_matrix(
                elev=elev, azim=azim,
                camera_distance=render.camera_distance, center=None,
            )
            pos_camera = transform_pos(r_mv, render.vtx_pos,  keepdim=True)
            pos_clip   = transform_pos(proj, pos_camera)

            rast_out, _ = render.raster_rasterize(
                pos_clip, render.pos_idx, resolution=(resolution, resolution)
            )
            visible = torch.clamp(rast_out[..., -1:], 0, 1)[0]   # H W 1

            uv, _ = render.raster_interpolate(
                render.vtx_uv[None, ...], rast_out, render.uv_idx
            )
            uv_norm = uv[0] * 2.0 - 1.0          # H W 2,  [-1, 1]
            uv_norm = uv_norm * visible           # zero background

            uv_maps[(azim, elev)] = uv_norm.to(device)
            masks  [(azim, elev)] = visible.to(device)

    return uv_maps, masks


# ---------------------------------------------------------------------------
# Differentiable render via grid_sample
# ---------------------------------------------------------------------------

def _render_from_texture(tex, uv_map, mask):
    """
    tex    : (1, 3, Ht, Wt) float [0,1]
    uv_map : (H, W, 2) float [-1,1]
    mask   : (H, W, 1) float {0,1}
    Returns (1, 3, H, W) white-background composite in [0,1].
    """
    grid     = uv_map.unsqueeze(0)
    rendered = F.grid_sample(tex, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)
    mask_4d  = mask.permute(2, 0, 1).unsqueeze(0)
    return rendered * mask_4d + (1.0 - mask_4d)   # white background


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float_np(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        t = arr.detach().cpu().float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        return t.numpy().astype(np.float32)
    arr = np.asarray(arr)
    return (arr / 255.0).astype(np.float32) if arr.dtype == np.uint8 else arr.astype(np.float32)


def _save_img(arr: np.ndarray, path: str):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SkinTextureRefiner(nn.Module):
    """
    Refines a baked UV texture using a frozen StyleGAN2-FFHQ discriminator.

    The discriminator score on rendered views acts as a realism signal.
    Only the texture pixels are optimised — D is fully frozen.

    Parameters
    ----------
    D_ckpt_path : str
        Path to the StyleGAN2-ADA-PyTorch FFHQ .pkl checkpoint.
    real_data_dir : str
        Unused (kept for API compatibility).
    num_steps : int
        Texture optimisation steps.
    lr_tex : float
        Adam learning-rate for texture.
    lr_D : float
        Unused (kept for API compatibility).
    lambda_adv : float
        Final adversarial loss weight (ramped up from 0 over
        ``adv_rampup_steps`` to prevent early texture shock).
    lambda_id : float
        Identity regularisation weight (MSE vs original texture).
    lambda_tv : float
        Total-variation smoothness weight.
    adv_rampup_steps : int
        Steps over which lambda_adv linearly ramps to its target.
    grad_clip : float
        Gradient-norm clipping for texture parameters.
    refine_resolution : int
        Render resolution during optimisation.
    num_views_per_step : int
        Viewpoints sampled each step.
    """

    def __init__(
        self,
        D_ckpt_path: str,
        real_data_dir: str     = "",     # unused
        num_steps: int         = 150,
        lr_tex: float          = 0.001,
        lr_D: float            = 0.0,    # unused
        lambda_adv: float      = 0.15,
        lambda_id: float       = 50.0,
        lambda_tv: float       = 0.5,
        adv_rampup_steps: int  = 50,
        grad_clip: float       = 0.1,
        max_delta: float       = 0.04,   # max per-pixel change from original
        refine_resolution: int = 512,
        num_views_per_step: int = 4,
    ):
        super().__init__()
        self.D_ckpt_path        = D_ckpt_path
        self.num_steps          = num_steps
        self.lr_tex             = lr_tex
        self.lambda_adv         = lambda_adv
        self.lambda_id          = lambda_id
        self.lambda_tv          = lambda_tv
        self.adv_rampup_steps   = adv_rampup_steps
        self.grad_clip          = grad_clip
        self.max_delta          = max_delta
        self.refine_resolution  = refine_resolution
        self.num_views_per_step = num_views_per_step

    # ------------------------------------------------------------------

    def forward(
        self,
        texture,
        texture_mr,
        render,
        reference_images: Optional[List[Image.Image]] = None,
        debug_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine albedo and MR textures.
        Returns (refined_albedo, refined_mr) as float32 numpy [0,1].
        """
        device = "cuda"
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # ---- Frozen discriminator ----------------------------------------
        D = StyleGAN2Discriminator(self.D_ckpt_path, device=device)
        # D is frozen inside StyleGAN2Discriminator.__init__

        # ---- UV maps --------------------------------------------------------
        print("[SkinRefiner] Precomputing UV maps …")
        viewpoints = [
            (azim, elev)
            for azim in range(0, 360, 30)
            for elev in [-20, 0, 20]
        ]
        uv_maps, masks = _extract_uv_maps(render, viewpoints, self.refine_resolution)
        vp_keys = list(uv_maps.keys())

        # ---- Texture parameters -------------------------------------------
        def _to_param(arr) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                t = arr.float().detach().to(device)
            else:
                arr = np.asarray(arr)
                t = torch.tensor(
                    arr / 255.0 if arr.dtype == np.uint8 else arr,
                    dtype=torch.float32, device=device,
                )
            if t.dim() == 3:
                t = t.permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(self.refine_resolution, self.refine_resolution),
                              mode="bilinear", align_corners=False)
            return t.requires_grad_(True)

        if isinstance(texture, torch.Tensor):
            orig_h, orig_w = texture.shape[0], texture.shape[1]
        else:
            orig_h, orig_w = np.asarray(texture).shape[:2]

        tex         = _to_param(texture)
        tex_mr      = _to_param(texture_mr)
        tex_orig    = tex.detach().clone()
        tex_mr_orig = tex_mr.detach().clone()

        optimizer_tex = torch.optim.Adam([tex, tex_mr], lr=self.lr_tex, betas=(0.9, 0.999))

        # ---- Debug helpers --------------------------------------------------
        debug_vps = [k for k in [(0, 0), (90, 0), (180, 0), (270, 0)] if k in uv_maps]

        def _save_views(t: torch.Tensor, tag: str):
            if debug_dir is None:
                return
            with torch.no_grad():
                for k in debug_vps:
                    img = _render_from_texture(t, uv_maps[k], masks[k])
                    img_np = img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    _save_img(img_np, os.path.join(debug_dir,
                              f"view_{tag}_az{k[0]:03d}_el{k[1]:+d}.png"))

        if debug_dir:
            _save_img(_to_float_np(tex_orig),    os.path.join(debug_dir, "texture_input.png"))
            _save_img(_to_float_np(tex_mr_orig), os.path.join(debug_dir, "texture_mr_input.png"))
            _save_views(tex_orig, "before")
            print(f"[SkinRefiner] Debug images → {debug_dir}")

        # ---- Optimisation loop ----------------------------------------------
        print(f"[SkinRefiner] Starting optimisation ({self.num_steps} steps) …")

        with torch.enable_grad():
            for step in range(self.num_steps):

                adv_w = self.lambda_adv * min(1.0, (step + 1) / max(1, self.adv_rampup_steps))

                optimizer_tex.zero_grad()

                sel = random.sample(vp_keys, self.num_views_per_step)
                renders = torch.cat([
                    _render_from_texture(tex, uv_maps[k], masks[k]) for k in sel
                ], dim=0)                                 # N 3 H W

                loss_G  = adv_w * adversarial_loss_G(D, renders)
                loss_id = self.lambda_id * (
                    F.mse_loss(tex, tex_orig) + F.mse_loss(tex_mr, tex_mr_orig)
                )
                loss_tv = self.lambda_tv * (total_variation(tex) + total_variation(tex_mr))
                loss    = loss_G + loss_id + loss_tv

                loss.backward()
                nn.utils.clip_grad_norm_([tex, tex_mr], self.grad_clip)
                optimizer_tex.step()

                with torch.no_grad():
                    # Per-pixel change clamp: prevents adversarial hallucinations
                    # from building up over successive steps.
                    tex.clamp_(tex_orig - self.max_delta, tex_orig + self.max_delta)
                    tex.clamp_(0.0, 1.0)
                    tex_mr.clamp_(tex_mr_orig - self.max_delta, tex_mr_orig + self.max_delta)
                    tex_mr.clamp_(0.0, 1.0)

                if (step + 1) % 10 == 0:
                    print(
                        f"[SkinRefiner] step {step + 1:3d}/{self.num_steps} | "
                        f"loss_G={loss_G.item():+.4f}  "
                        f"loss_id={loss_id.item():.4f}  "
                        f"loss_tv={loss_tv.item():.5f}  "
                        f"adv_w={adv_w:.3f}"
                    )

        print("[SkinRefiner] Optimisation complete.")

        if debug_dir:
            _save_views(tex.detach(), "after")
            _save_img(_to_float_np(tex.detach()),    os.path.join(debug_dir, "texture_output.png"))
            _save_img(_to_float_np(tex_mr.detach()), os.path.join(debug_dir, "texture_mr_output.png"))

        def _to_numpy(t: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                t_up = F.interpolate(t, size=(orig_h, orig_w),
                                     mode="bilinear", align_corners=False)
            return t_up.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        return _to_numpy(tex), _to_numpy(tex_mr)
