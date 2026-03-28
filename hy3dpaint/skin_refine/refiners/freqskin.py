"""skin_refine/refiners/freqskin.py — Frequency-decomposed skin texture refinement.
===================================================================================
Novel UV-space refinement that bypasses the lossy render→bake loop entirely.

Method (FreqSkin):
  1. Decompose the UV texture into a Laplacian pyramid.
  2. Keep low-frequency bands (color, tone, shading) from the original.
  3. Synthesize realistic high-frequency skin detail (pores, micro-wrinkles)
     using SD 1.5 + ControlNet Tile, conditioned on the blurred texture.
  4. Reconstruct the final texture by combining original low-freq + synthesized high-freq.

Advantages over screen-space refiners (GFPGAN, CodeFormer, SD):
  - No render→bake quality loss (works directly on the UV map)
  - Full UV resolution preserved
  - No view-seam artifacts
  - Generates rather than restores detail

Debug outputs (written to debug_dir when provided):
  sd_input.png                  — full UV texture fed to SD
  sd_synthesized_full.png       — full SD output before pyramid blending
  tile_coverage.png             — tile map: green=processed, yellow=skipped(non-skin), black=empty
  skin_ratio_heatmap.png        — per-tile skin ratio as a heatmap (blue→red)
  band_orig_L{i}.png            — original Laplacian bands (normalized)
  band_orig_residual.png        — coarsest Gaussian level
  band_synth_L{i}.png           — synthesized Laplacian bands
  band_diff_L{i}.png            — difference (synth - orig) per high-freq band
  recon_split_{k}.png           — reconstructed texture using only bands 0..k from SD
  freqskin_output.png           — final refined texture
  comparison.png                — [original | SD_raw | final] side-by-side
  tiles/tile_{idx}_input.png    — per-tile SD input
  tiles/tile_{idx}_output.png   — per-tile SD output
"""

import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_SD15_PRETRAINED = "SG161222/Realistic_Vision_V6.0_B1_noVAE" #"marcagve18/baby-face-generation"
_SD15_VAE        = "stabilityai/sd-vae-ft-ema"
_SD15_CONTROLNET = "lllyasviel/control_v11f1e_sd15_tile"


# ---------------------------------------------------------------------------
# Laplacian Pyramid Utilities
# ---------------------------------------------------------------------------

def _gaussian_blur(tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """Apply Gaussian blur to a (H, W, C) tensor. Returns same shape."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    C = tensor.shape[-1]
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    x = tensor.permute(2, 0, 1).unsqueeze(0)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, kernel, groups=C)
    return out.squeeze(0).permute(1, 2, 0)


def _downsample(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, scale_factor=0.5, mode="area")
    return x.squeeze(0).permute(1, 2, 0)


def _upsample(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    x = tensor.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0)


def build_laplacian_pyramid(
    texture: torch.Tensor,
    num_levels: int = 4,
    blur_kernel: int = 5,
    blur_sigma: float = 1.5,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Decompose (H, W, C) texture into a Laplacian pyramid.

    Returns
    -------
    laplacian_bands : list of (H_k, W_k, C) tensors  (L_0 .. L_{N-1})
    residual : (H_N, W_N, C) tensor
    """
    gaussians = [texture]
    current = texture
    for _ in range(num_levels):
        blurred = _gaussian_blur(current, blur_kernel, blur_sigma)
        down = _downsample(blurred)
        gaussians.append(down)
        current = down

    laplacian_bands = []
    for k in range(num_levels):
        g_k = gaussians[k]
        g_k1_up = _upsample(gaussians[k + 1], g_k.shape[0], g_k.shape[1])
        laplacian_bands.append(g_k - g_k1_up)

    return laplacian_bands, gaussians[num_levels]


def reconstruct_from_pyramid(
    laplacian_bands: List[torch.Tensor],
    residual: torch.Tensor,
) -> torch.Tensor:
    current = residual
    for band in reversed(laplacian_bands):
        current = _upsample(current, band.shape[0], band.shape[1]) + band
    return current


# ---------------------------------------------------------------------------
# Tiled Processing Utilities
# ---------------------------------------------------------------------------

def _tile_coordinates(
    H: int, W: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    step = tile_size - overlap
    tiles = []
    for y0 in range(0, H, step):
        y1 = min(y0 + tile_size, H)
        if y1 - y0 < tile_size and y0 > 0:
            y0 = max(0, y1 - tile_size)
        for x0 in range(0, W, step):
            x1 = min(x0 + tile_size, W)
            if x1 - x0 < tile_size and x0 > 0:
                x0 = max(0, x1 - tile_size)
            tiles.append((y0, x0, y1, x1))
    return list(dict.fromkeys(tiles))


def _create_blend_mask(h: int, w: int, overlap: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(h, w, device=device)
    if overlap <= 0:
        return mask
    ramp = torch.linspace(0, 1, overlap, device=device)
    mask[:overlap, :] *= ramp[:, None]
    mask[-overlap:, :] *= ramp.flip(0)[:, None]
    mask[:, :overlap] *= ramp[None, :]
    mask[:, -overlap:] *= ramp.flip(0)[None, :]
    return mask


# ---------------------------------------------------------------------------
# Debug Utilities
# ---------------------------------------------------------------------------

def _save_png(arr_or_tensor, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(arr_or_tensor, torch.Tensor):
        arr_or_tensor = arr_or_tensor.detach().cpu().numpy()
    arr = np.clip(arr_or_tensor * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_band(band: torch.Tensor, path: str):
    """Save a Laplacian band (can be negative) centered at 0.5."""
    vis = ((band.detach().cpu().numpy() + 0.5) * 255).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(vis).save(path)


def _colormap_heatmap(values: np.ndarray) -> np.ndarray:
    """Map [0,1] scalar array to blue→green→red uint8 RGB via a simple jet-like ramp."""
    r = np.clip(1.5 - np.abs(values * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(values * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(values * 4 - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _save_tile_coverage(
    H: int, W: int,
    tiles: List[Tuple[int, int, int, int]],
    tile_status: List[str],   # "processed" | "skipped_skin" | "skipped_empty"
    path: str,
    skin_ratios: Optional[List[float]] = None,
    threshold: float = 0.4,
):
    """Save a coverage map and a skin-ratio heatmap side by side."""
    coverage = np.zeros((H, W, 3), dtype=np.uint8)
    heatmap_vals = np.zeros((H, W), dtype=np.float32)

    COLOR = {
        "processed":     (0,   200, 80),    # green
        "skipped_skin":  (220, 180, 0),     # yellow
        "skipped_empty": (30,  30,  30),    # near-black
    }

    for idx, ((y0, x0, y1, x1), status) in enumerate(zip(tiles, tile_status)):
        coverage[y0:y1, x0:x1] = COLOR[status]
        if skin_ratios is not None:
            heatmap_vals[y0:y1, x0:x1] = skin_ratios[idx]

    # Draw tile borders
    cov_img = Image.fromarray(coverage)
    draw = ImageDraw.Draw(cov_img)
    for (y0, x0, y1, x1) in tiles:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255), width=1)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cov_img.save(path)

    # Save heatmap alongside
    if skin_ratios is not None:
        heat_rgb = _colormap_heatmap(heatmap_vals)
        # Draw threshold line indicator in title area — just save the heatmap
        heat_img = Image.fromarray(heat_rgb)
        heatmap_path = path.replace("tile_coverage", "skin_ratio_heatmap")
        heat_img.save(heatmap_path)
        logger.info(f"[FreqSkin] Skin ratio heatmap → {heatmap_path}")

    logger.info(f"[FreqSkin] Tile coverage map → {path}")


def _save_comparison(orig: torch.Tensor, sd_raw: torch.Tensor, final: torch.Tensor, path: str):
    """Save [original | SD raw output | final reconstructed] side by side."""
    sep = np.ones((orig.shape[0], 6, 3), dtype=np.uint8) * 180
    o = (orig.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    s = (sd_raw.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    f = (final.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    combined = np.concatenate([o, sep, s, sep, f], axis=1)
    img = Image.fromarray(combined)

    # Simple text labels via drawing (no font needed)
    draw = ImageDraw.Draw(img)
    W3 = orig.shape[1]
    for label, x in [("Original", 4), ("SD Raw", W3 + 10), ("Final", W3 * 2 + 16)]:
        draw.rectangle([x, 0, x + len(label) * 7 + 4, 14], fill=(0, 0, 0))
        draw.text((x + 2, 1), label, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
    logger.info(f"[FreqSkin] Comparison image → {path}")


# ---------------------------------------------------------------------------
# FreqSkinRefiner
# ---------------------------------------------------------------------------

class FreqSkinRefiner(BaseSkinRefiner):
    """Frequency-decomposed UV-space skin texture refiner using SD 1.5 + ControlNet Tile.

    Parameters
    ----------
    num_levels : int
        Laplacian pyramid levels (default: 4).
    freq_split : int
        Bands 0..(freq_split-1) replaced by SD synthesis; rest kept from original.
        Default: 2.
    tile_size : int
        Tile size for SD processing. SD 1.5 is trained at 512 — use 512 or 768.
        Default: 512.
    tile_overlap : int
        Overlap between tiles in pixels (default: 128).
    strength : float
        img2img denoising strength (default: 0.12). Keep low to preserve structure.
    controlnet_scale : float
        ControlNet conditioning scale (default: 1.0).
    guidance_scale : float
        CFG scale (default: 5.0).
    num_inference_steps : int
        Diffusion sampling steps (default: 25).
    prompt : str
        Positive prompt for skin detail synthesis.
    negative_prompt : str, optional
        Negative prompt. Defaults to standard anti-artifact prompt.
    band_weights : list of float
        Per-band blend weight for synthesized high-freq bands.
    seed : int
        RNG seed (default: 42).
    skin_ratio_threshold : float
        Min fraction of skin pixels required to process a tile (default: 0.4).
        Tiles below this (eyes, lips, hair) are passed through unchanged.
    pretrained_path : str
        SD 1.5 checkpoint.
    vae_path : str
        VAE checkpoint.
    controlnet_path : str
        ControlNet checkpoint.
    device : str
        Device (default: 'cuda').
    """

    def __init__(
        self,
        num_levels: int = 4,
        freq_split: int = 2,
        tile_size: int = 512,
        tile_overlap: int = 128,
        strength: float = 0.40,
        controlnet_scale: float = 1.0,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 25,
        prompt: str = "extreme close-up of baby skin, photorealistic skin pores, subsurface scattering, natural skin texture, 8k texture map",
        negative_prompt: Optional[str] = None,
        band_weights: Optional[List[float]] = None,
        seed: int = 42,
        skin_ratio_threshold: float = 0.4,
        pretrained_path: str = _SD15_PRETRAINED,
        vae_path: str = _SD15_VAE,
        controlnet_path: str = _SD15_CONTROLNET,
        device: str = "cuda",
    ):
        self.num_levels = num_levels
        self.freq_split = freq_split
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.strength = strength
        self.controlnet_scale = controlnet_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.prompt = prompt
        self.negative_prompt = negative_prompt or (
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
            "text, worst quality, low quality, jpeg artifacts, ugly, blurry, bad anatomy, disfigured"
        )
        self.band_weights = band_weights or [0.8] * freq_split
        self.seed = seed
        self.skin_ratio_threshold = skin_ratio_threshold
        self.pretrained_path = pretrained_path
        self.vae_path = vae_path
        self.controlnet_path = controlnet_path
        self._device = device
        self._pipeline = None

    @property
    def name(self) -> str:
        return "FreqSkin-SD15"

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from diffusers import ControlNetModel, AutoencoderKL
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        from diffusers.schedulers import EulerDiscreteScheduler

        logger.info("[FreqSkin] Loading ControlNet Tile…")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path, torch_dtype=torch.float16
        )

        logger.info("[FreqSkin] Loading VAE…")
        vae = AutoencoderKL.from_pretrained(self.vae_path, torch_dtype=torch.float16)

        logger.info(f"[FreqSkin] Loading SD 1.5 ({self.pretrained_path})…")
        self._pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.pretrained_path,
            vae=vae,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        self._pipeline.to(self._device)
        self._pipeline.safety_checker = None
        self._pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self._pipeline.scheduler.config
        )
        self._pipeline.set_progress_bar_config(disable=True)
        logger.info("[FreqSkin] SD 1.5 pipeline ready.")

    def _unload_pipeline(self):
        if self._pipeline is not None:
            self._pipeline.to("cpu")
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Skin detection
    # ------------------------------------------------------------------

    @staticmethod
    def _skin_pixel_ratio(tile_tensor: torch.Tensor) -> float:
        """Estimate fraction of skin-like pixels using HSV thresholding.

        Skin pixels in HSV (normalized [0,1]):
          H ∈ [0.0, 0.10]  (red–orange–yellow)
          S ∈ [0.10, 0.90]  (not grey, not fully saturated)
          V ∈ [0.20, 0.95]  (not black, not blown out)

        Non-skin regions (eyes, lips, dark hair) fall outside these ranges.
        """
        rgb = tile_tensor.cpu().float()
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        Cmax, _ = rgb.max(dim=-1)
        Cmin, _ = rgb.min(dim=-1)
        delta = Cmax - Cmin

        V = Cmax
        S = torch.where(Cmax > 1e-6, delta / Cmax, torch.zeros_like(Cmax))

        H = torch.zeros_like(Cmax)
        mask_r = (Cmax == R) & (delta > 1e-6)
        mask_g = (Cmax == G) & (delta > 1e-6)
        mask_b = (Cmax == B) & (delta > 1e-6)
        H[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6.0
        H[mask_g] = (B[mask_g] - R[mask_g]) / delta[mask_g] + 2.0
        H[mask_b] = (R[mask_b] - G[mask_b]) / delta[mask_b] + 4.0
        H = H / 6.0

        skin = (
            ((H < 0.10) | (H > 0.93)) &
            (S > 0.10) & (S < 0.90) &
            (V > 0.20) & (V < 0.95)
        )
        return skin.float().mean().item()

    # ------------------------------------------------------------------
    # Tiled SD processing
    # ------------------------------------------------------------------

    def _process_tile(self, tile_pil: Image.Image) -> Image.Image:
        """Run SD 1.5 ControlNet Tile img2img on a single tile."""
        w, h = tile_pil.size
        w8, h8 = (w // 8) * 8, (h // 8) * 8
        if (w8, h8) != (w, h):
            tile_pil = tile_pil.resize((w8, h8), Image.LANCZOS)

        generator = torch.Generator(device=self._device).manual_seed(self.seed)

        result = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=tile_pil,
            control_image=tile_pil,
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

    def _process_texture_tiled(
        self, texture: torch.Tensor, debug_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Process full UV texture with tiled SD, return synthesized texture."""
        H, W, C = texture.shape
        tile_sz = self.tile_size
        overlap = self.tile_overlap

        tiles = _tile_coordinates(H, W, tile_sz, overlap)
        logger.info(f"[FreqSkin] Processing {len(tiles)} tiles ({tile_sz}px, overlap={overlap})…")

        result = torch.zeros_like(texture)
        weight = torch.zeros(H, W, 1, device=texture.device)

        # Track per-tile status for debug maps
        tile_status: List[str] = []
        skin_ratios: List[float] = []

        for idx, (y0, x0, y1, x1) in enumerate(tiles):
            th, tw = y1 - y0, x1 - x0
            tile_tensor = texture[y0:y1, x0:x1, :]

            # Skip empty tiles (UV background)
            if tile_tensor.mean() < 0.01:
                tile_status.append("skipped_empty")
                skin_ratios.append(0.0)
                continue

            # Skip tiles with insufficient skin pixels
            skin_ratio = self._skin_pixel_ratio(tile_tensor)
            skin_ratios.append(skin_ratio)

            if skin_ratio < self.skin_ratio_threshold:
                logger.debug(
                    f"[FreqSkin] Tile {idx+1}/{len(tiles)} skipped "
                    f"(skin_ratio={skin_ratio:.2f} < {self.skin_ratio_threshold})."
                )
                tile_status.append("skipped_skin")
                bmask = _create_blend_mask(th, tw, overlap, texture.device).unsqueeze(-1)
                result[y0:y1, x0:x1, :] += tile_tensor * bmask
                weight[y0:y1, x0:x1, :] += bmask
                continue

            tile_status.append("processed")
            tile_np = (tile_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            tile_pil = Image.fromarray(tile_np)

            needs_resize = (th != tile_sz or tw != tile_sz)
            if needs_resize:
                tile_pil = tile_pil.resize((tile_sz, tile_sz), Image.LANCZOS)

            refined_pil = self._process_tile(tile_pil)

            if debug_dir:
                tile_dir = os.path.join(debug_dir, "tiles")
                os.makedirs(tile_dir, exist_ok=True)
                tile_pil.save(os.path.join(tile_dir, f"tile_{idx:03d}_sr{skin_ratio:.2f}_input.png"))
                refined_pil.save(os.path.join(tile_dir, f"tile_{idx:03d}_sr{skin_ratio:.2f}_output.png"))

            if needs_resize:
                refined_pil = refined_pil.resize((tw, th), Image.LANCZOS)

            refined_tensor = torch.from_numpy(
                np.array(refined_pil).astype(np.float32) / 255.0
            ).to(texture.device)

            bmask = _create_blend_mask(th, tw, overlap, texture.device).unsqueeze(-1)
            result[y0:y1, x0:x1, :] += refined_tensor * bmask
            weight[y0:y1, x0:x1, :] += bmask

            if (idx + 1) % 5 == 0 or idx == len(tiles) - 1:
                logger.info(f"[FreqSkin] Tile {idx+1}/{len(tiles)} done.")

        result = result / weight.clamp(min=1e-8)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            _save_png(result, os.path.join(debug_dir, "sd_synthesized_full.png"))
            _save_tile_coverage(
                H, W, tiles, tile_status,
                path=os.path.join(debug_dir, "tile_coverage.png"),
                skin_ratios=skin_ratios,
                threshold=self.skin_ratio_threshold,
            )
            n_proc = tile_status.count("processed")
            n_skip = tile_status.count("skipped_skin")
            n_empty = tile_status.count("skipped_empty")
            logger.info(
                f"[FreqSkin] Tile summary: {n_proc} processed, "
                f"{n_skip} skipped (non-skin), {n_empty} skipped (empty)."
            )

        return result

    # ------------------------------------------------------------------
    # Core UV-space refinement
    # ------------------------------------------------------------------

    def refine_uv(
        self,
        texture: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        debug_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """Refine a UV texture using frequency decomposition + SD 1.5.

        Parameters
        ----------
        texture : (H, W, 3) float tensor [0, 1]
        mask : (H, W) or (H, W, 1) float tensor, optional  (1=process, 0=skip)
        debug_dir : str, optional

        Returns
        -------
        refined : (H, W, 3) float tensor [0, 1]
        """
        logger.info(
            f"[FreqSkin] UV-space frequency refinement "
            f"(levels={self.num_levels}, split={self.freq_split}, strength={self.strength})…"
        )

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # Step 1: Decompose original texture
        bands_orig, residual = build_laplacian_pyramid(texture, self.num_levels)

        if debug_dir:
            for i, band in enumerate(bands_orig):
                _save_band(band, os.path.join(debug_dir, f"band_orig_L{i}.png"))
            _save_png(residual, os.path.join(debug_dir, "band_orig_residual.png"))
            _save_png(texture.clamp(0, 1), os.path.join(debug_dir, "sd_input.png"))

        # Step 2: Run SD on the original texture to synthesize high-freq detail
        self._load_pipeline()
        synthesized = self._process_texture_tiled(texture.clamp(0, 1), debug_dir)
        self._unload_pipeline()

        # Step 3: Decompose the synthesized texture
        bands_synth, _ = build_laplacian_pyramid(synthesized, self.num_levels)

        if debug_dir:
            for i in range(self.freq_split):
                _save_band(bands_synth[i], os.path.join(debug_dir, f"band_synth_L{i}.png"))
                # Band difference: what SD is actually adding
                diff = bands_synth[i] - bands_orig[i]
                _save_band(diff, os.path.join(debug_dir, f"band_diff_L{i}.png"))

        # Step 4: Reconstruct — original low-freq + synthesized high-freq
        final_bands = []
        for i in range(self.num_levels):
            if i < self.freq_split:
                w = self.band_weights[i] if i < len(self.band_weights) else 0.8
                final_bands.append((1 - w) * bands_orig[i] + w * bands_synth[i])
            else:
                final_bands.append(bands_orig[i])

        refined = reconstruct_from_pyramid(final_bands, residual).clamp(0, 1)

        # Debug: per-split reconstructions (ablation)
        if debug_dir:
            for split_k in range(1, self.freq_split + 1):
                ablation_bands = []
                for i in range(self.num_levels):
                    if i < split_k:
                        w = self.band_weights[i] if i < len(self.band_weights) else 0.8
                        ablation_bands.append((1 - w) * bands_orig[i] + w * bands_synth[i])
                    else:
                        ablation_bands.append(bands_orig[i])
                recon = reconstruct_from_pyramid(ablation_bands, residual).clamp(0, 1)
                _save_png(recon, os.path.join(debug_dir, f"recon_split_{split_k}.png"))

        # Step 5: Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(texture.device).float()
            refined = mask * refined + (1 - mask) * texture

        if debug_dir:
            _save_png(refined, os.path.join(debug_dir, "freqskin_output.png"))
            _save_comparison(
                texture.clamp(0, 1), synthesized, refined,
                os.path.join(debug_dir, "comparison.png"),
            )

        logger.info("[FreqSkin] Frequency refinement complete.")
        return refined

    # ------------------------------------------------------------------
    # BaseSkinRefiner fallback (screen-space mode)
    # ------------------------------------------------------------------

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Fallback: refine a single rendered view via SD img2img."""
        self._load_pipeline()
        try:
            original_size = image.size
            tile_pil = image.resize((self.tile_size, self.tile_size), Image.LANCZOS)
            refined = self._process_tile(tile_pil)
            return refined.resize(original_size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"[FreqSkin] restore() failed: {e}")
            return None

    def to(self, device: str):
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self
