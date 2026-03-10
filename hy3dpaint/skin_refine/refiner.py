"""
SkinTextureRefiner
==================
Post-processing module for the Hunyuan3D-2.1 paint pipeline.

Takes the baked UV texture (which can look plastic on faces), optimises its
pixel values using an adversarial (WGAN-GP) loss so that renders of the
textured mesh look like realistic skin.

Usage (inserted in textureGenPipeline.py after inpainting):

    texture, texture_mr = self.models["skin_refiner"](
        texture=texture,
        texture_mr=texture_mr,
        render=self.render,
        reference_images=image_style,
    )
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .discriminator import StyleGAN2Discriminator
from .losses import adversarial_loss_D, adversarial_loss_G, total_variation

# Imported lazily from the renderer package (same pattern used in MeshRender.py)
try:
    from DifferentiableRenderer.camera_utils import get_mv_matrix, transform_pos
except ImportError:
    from camera_utils import get_mv_matrix, transform_pos


# ---------------------------------------------------------------------------
# Simple image folder dataset (no torchvision dependency required)
# ---------------------------------------------------------------------------

class _ImageFolderDataset(Dataset):
    """Load all images from a directory recursively."""

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str, size: int = 512):
        self.size = size
        self.paths: List[str] = []
        for dirpath, _, fnames in os.walk(root):
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() in self.EXTS:
                    self.paths.append(os.path.join(dirpath, fn))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.size, self.size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0          # H W 3  [0,1]
        return torch.tensor(arr).permute(2, 0, 1)               # 3 H W


# ---------------------------------------------------------------------------
# UV-map extraction helpers
# ---------------------------------------------------------------------------

def _extract_uv_maps(
    render,
    viewpoints: List[Tuple[float, float]],
    resolution: int = 512,
) -> Tuple[Dict, Dict]:
    """
    Precompute per-viewpoint UV coordinate maps and visibility masks.

    Uses the renderer's rasterizer directly (same pattern as MeshRender.back_project)
    to interpolate UV vertex attributes at each visible pixel.

    Returns:
        uv_maps:  dict (azim, elev) -> (H, W, 2) tensor in [-1, 1], device=cuda
        masks:    dict (azim, elev) -> (H, W, 1) tensor in {0, 1},   device=cuda
    """
    device = render.device
    proj = render.camera_proj_mat

    uv_maps: Dict = {}
    masks: Dict = {}

    with torch.no_grad():
        for azim, elev in viewpoints:
            r_mv = get_mv_matrix(
                elev=elev,
                azim=azim,
                camera_distance=render.camera_distance,
                center=None,
            )
            pos_camera = transform_pos(r_mv, render.vtx_pos, keepdim=True)
            pos_clip = transform_pos(proj, pos_camera)

            rast_out, _ = render.raster_rasterize(pos_clip, render.pos_idx, resolution=(resolution, resolution))
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0]  # H W 1

            # Interpolate UV coords at each pixel  (vtx_uv ∈ [0,1])
            uv, _ = render.raster_interpolate(render.vtx_uv[None, ...], rast_out, render.uv_idx)
            uv = uv[0]  # H W 2,  values in [0, 1]

            # grid_sample expects coords in [-1, 1]
            uv_norm = uv * 2.0 - 1.0                     # H W 2
            uv_norm = uv_norm * visible_mask              # zero out background

            uv_maps[(azim, elev)] = uv_norm.to(device)   # H W 2
            masks[(azim, elev)] = visible_mask.to(device) # H W 1

    return uv_maps, masks


# ---------------------------------------------------------------------------
# Differentiable render via grid_sample
# ---------------------------------------------------------------------------

def _render_from_texture(
    tex: torch.Tensor,
    uv_map: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Render a view from a texture using precomputed UV coordinates.

    Args:
        tex:    (1, 3, H_tex, W_tex)  — differentiable, values in [0, 1]
        uv_map: (H, W, 2)             — fixed, detached, values in [-1, 1]
        mask:   (H, W, 1)             — fixed, detached, binary

    Returns:
        (1, 3, H, W) differentiable rendered image in [0, 1]
    """
    grid = uv_map.unsqueeze(0)                          # 1 H W 2
    rendered = F.grid_sample(
        tex, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )                                                   # 1 3 H W
    mask_4d = mask.permute(2, 0, 1).unsqueeze(0)        # 1 1 H W
    return rendered * mask_4d


# ---------------------------------------------------------------------------
# Main refiner class
# ---------------------------------------------------------------------------

class SkinTextureRefiner(nn.Module):
    """
    Refines a baked UV texture using adversarial training so that
    renderings look like realistic skin.

    Parameters
    ----------
    D_ckpt_path : str
        Path to StyleGAN2-ADA-PyTorch FFHQ .pkl checkpoint.
    real_data_dir : str
        Directory of real reference images (512×512 baby-face crops).
    num_steps : int
        Optimisation iterations.
    lr_tex : float
        Learning-rate for texture parameters.
    lr_D : float
        Learning-rate for discriminator fine-tuning.
    lambda_gp : float
        Gradient-penalty weight.
    lambda_adv : float
        Adversarial loss weight.
    lambda_perc : float
        LPIPS perceptual loss weight.
    lambda_id : float
        Identity / reconstruction loss weight.
    lambda_tv : float
        Total-variation smoothness weight.
    num_views_per_step : int
        Viewpoints sampled each optimisation step.
    D_steps : int
        Discriminator updates per texture update.
    refine_resolution : int
        Resolution at which renders are generated (px).
    """

    def __init__(
        self,
        D_ckpt_path: str,
        real_data_dir: str,
        num_steps: int = 150,
        lr_tex: float = 0.001,
        lr_D: float = 1e-4,       # unused — D is frozen
        lambda_gp: float = 10.0,  # unused — no D training
        lambda_adv: float = 0.1,
        lambda_perc: float = 1.0,
        lambda_id: float = 5.0,
        lambda_tv: float = 0.01,
        num_views_per_step: int = 4,
        D_steps: int = 1,
        refine_resolution: int = 512,
    ):
        super().__init__()

        self.D_ckpt_path = D_ckpt_path
        self.real_data_dir = real_data_dir
        self.num_steps = num_steps
        self.lr_tex = lr_tex
        self.lr_D = lr_D
        self.lambda_gp = lambda_gp
        self.lambda_adv = lambda_adv
        self.lambda_perc = lambda_perc
        self.lambda_id = lambda_id
        self.lambda_tv = lambda_tv
        self.num_views_per_step = num_views_per_step
        self.D_steps = D_steps
        self.refine_resolution = refine_resolution

    # ------------------------------------------------------------------
    # Public entry point (callable like a function)
    # ------------------------------------------------------------------

    def forward(
        self,
        texture: np.ndarray,
        texture_mr: np.ndarray,
        render,
        reference_images: Optional[List[Image.Image]] = None,
        debug_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine albedo and metallic-roughness textures.

        Args:
            texture:          (H, W, 3) uint8 numpy array — albedo.
            texture_mr:       (H, W, 3) uint8 numpy array — metallic-roughness.
            render:           MeshRender instance (mesh already loaded).
            reference_images: List of PIL reference images (optional; used for
                              perceptual loss target if supplied).

        Returns:
            (refined_texture, refined_texture_mr) as uint8 numpy arrays.
        """
        device = "cuda"

        # ---- Debug helpers ------------------------------------------------
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

        def _save_debug_views(tex_t: torch.Tensor, tag: str, vp_keys_subset):
            if debug_dir is None:
                return
            with torch.no_grad():
                for k in vp_keys_subset:
                    rendered = _render_from_texture(tex_t, uv_maps[k], masks[k])
                    mask_4d = masks[k].permute(2, 0, 1).unsqueeze(0)
                    rendered_white = rendered + (1.0 - mask_4d)
                    img_np = (rendered_white.squeeze(0).permute(1, 2, 0)
                              .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    fname = f"view_{tag}_az{k[0]:03d}_el{k[1]:+d}.png"
                    Image.fromarray(img_np).save(os.path.join(debug_dir, fname))

        # ---- Discriminator (frozen — used as a fixed feature extractor) ---
        # Fine-tuning D alongside the texture causes GAN divergence within
        # 150 steps on a single texture.  Instead we freeze D and use it as
        # a stable adversarial signal, similar to perceptual loss.
        print("[SkinRefiner] Loading StyleGAN2 discriminator …")
        D = StyleGAN2Discriminator(self.D_ckpt_path, device=device)
        D.eval()
        for p in D.parameters():
            p.requires_grad_(False)

        # ---- Perceptual loss ----------------------------------------------
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

        # ---- Perceptual reference (first reference image if available) ----
        if reference_images is not None and len(reference_images) > 0:
            ref_pil = reference_images[0].resize(
                (self.refine_resolution, self.refine_resolution), Image.LANCZOS
            ).convert("RGB")
            ref_np = np.array(ref_pil, dtype=np.float32) / 255.0
            ref_tensor = (
                torch.tensor(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)
            )  # 1 3 H W
        else:
            ref_tensor = None

        # ---- Precompute UV maps -------------------------------------------
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
                if arr.dtype == np.uint8:
                    t = torch.tensor(arr / 255.0, dtype=torch.float32, device=device)
                else:
                    t = torch.tensor(arr, dtype=torch.float32, device=device)
            if t.dim() == 3:  # H W C → 1 C H W
                t = t.permute(2, 0, 1).unsqueeze(0)
            return t.requires_grad_(True)

        tex = _to_param(texture)
        tex_mr = _to_param(texture_mr)
        tex_orig = tex.clone().detach()
        tex_mr_orig = tex_mr.clone().detach()

        # Debug views: front, left, right, back at 0° elevation
        debug_vps = [(0, 0), (90, 0), (180, 0), (270, 0)]
        debug_vps = [k for k in debug_vps if k in uv_maps]

        if debug_dir is not None:
            # Save input texture
            _tex_np = (tex_orig.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(_tex_np).save(os.path.join(debug_dir, "texture_input.png"))
            # Save reference image
            if ref_tensor is not None:
                _ref_np = (ref_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(_ref_np).save(os.path.join(debug_dir, "ref_image.png"))
            # Save pre-optimisation renders
            _save_debug_views(tex_orig, "before", debug_vps)
            print(f"[SkinRefiner] Debug images will be saved to: {debug_dir}")

        optimizer_tex = torch.optim.Adam([tex, tex_mr], lr=self.lr_tex)

        # ---- Optimisation loop -------------------------------------------
        # torch.enable_grad() is required because the pipeline __call__ runs
        # under @torch.no_grad(), which would block all backward passes.
        print(f"[SkinRefiner] Starting optimisation ({self.num_steps} steps) …")
        with torch.enable_grad():
            for step in range(self.num_steps):

                optimizer_tex.zero_grad()

                sel = random.sample(vp_keys, self.num_views_per_step)
                loss: torch.Tensor = torch.tensor(0.0, device=device)

                for k in sel:
                    # White background matches the real reference images
                    rendered = _render_from_texture(tex, uv_maps[k], masks[k])       # 1 3 H W
                    mask_4d = masks[k].permute(2, 0, 1).unsqueeze(0)                  # 1 1 H W
                    rendered_white = rendered + (1.0 - mask_4d)                       # white bg

                    # Adversarial: frozen D as a fixed quality signal
                    loss = loss + self.lambda_adv * adversarial_loss_G(D, rendered_white)

                    # Perceptual: LPIPS against reference image
                    if ref_tensor is not None and self.lambda_perc > 0:
                        loss = loss + self.lambda_perc * lpips_fn(rendered_white, ref_tensor).mean()

                loss = loss / self.num_views_per_step

                # Identity: stay close to the original baked texture
                loss_id = F.mse_loss(tex, tex_orig) + F.mse_loss(tex_mr, tex_mr_orig)
                loss = loss + self.lambda_id * loss_id

                # Total variation: spatial smoothness
                loss_tv = total_variation(tex) + total_variation(tex_mr)
                loss = loss + self.lambda_tv * loss_tv

                loss.backward()
                optimizer_tex.step()

                with torch.no_grad():
                    tex.clamp_(0.0, 1.0)
                    tex_mr.clamp_(0.0, 1.0)

                if (step + 1) % 25 == 0:
                    print(
                        f"[SkinRefiner] step {step + 1}/{self.num_steps} | "
                        f"loss={loss.item():.4f}  loss_id={loss_id.item():.4f}"
                    )

        print("[SkinRefiner] Optimisation complete.")

        # ---- Convert back to float numpy [0, 1] ---------------------------------
        # set_texture(force_set=True) calls torch.from_numpy() without /255,
        # so we must return float [0,1] not uint8 [0,255].
        def _to_numpy(t: torch.Tensor) -> np.ndarray:
            return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        if debug_dir is not None:
            _save_debug_views(tex.detach(), "after", debug_vps)
            _tex_out_u8 = (_to_numpy(tex) * 255).astype(np.uint8)
            Image.fromarray(_tex_out_u8).save(os.path.join(debug_dir, "texture_output.png"))

        return _to_numpy(tex), _to_numpy(tex_mr)
