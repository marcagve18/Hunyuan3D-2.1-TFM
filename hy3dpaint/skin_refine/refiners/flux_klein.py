"""skin_refine/refiners/flux_klein.py — Tiled UV-space refiner using Flux Klein 4B.
==================================================================================
Processes the UV texture directly (no render→bake loop) using overlapping tiles
refined by Flux 2 Klein 4B with an optional LoRA.  Tiles are large (1024px) to
give the model enough spatial context, with feathered overlap blending to
eliminate seam artifacts.

The refiner exposes ``refine_uv()`` so the orchestrator dispatches it in
UV-space mode, bypassing the multi-view render→restore→bake loop entirely.
"""

import os
import logging
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
_DEFAULT_LORA = None


# ---------------------------------------------------------------------------
# Tile utilities (shared logic with freqskin but tuned for large tiles)
# ---------------------------------------------------------------------------

def _tile_coords(H: int, W: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Generate (y0, x0, y1, x1) tile coordinates with overlap."""
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


def _feather_mask(h: int, w: int, overlap: int, device: torch.device) -> torch.Tensor:
    """Create a (H, W) feathered blend mask: 1 in center, ramps to 0 at edges."""
    mask = torch.ones(h, w, device=device)
    if overlap <= 0:
        return mask
    ramp = torch.linspace(0, 1, overlap, device=device)
    mask[:overlap, :] *= ramp[:, None]
    mask[-overlap:, :] *= ramp.flip(0)[:, None]
    mask[:, :overlap] *= ramp[None, :]
    mask[:, -overlap:] *= ramp.flip(0)[None, :]
    return mask


def _is_empty(tile: torch.Tensor, threshold: float = 0.02) -> bool:
    """True if the tile is UV background (near-black)."""
    return tile.mean().item() < threshold


# ---------------------------------------------------------------------------
# FluxKleinTiledRefiner
# ---------------------------------------------------------------------------

class FluxKleinTiledRefiner(BaseSkinRefiner):
    """UV-space tiled refiner using Flux 2 Klein 4B.

    Parameters
    ----------
    model : str
        HuggingFace model ID for Klein 4B.
    lora_path : str, optional
        Path to a LoRA safetensors file to fuse into the model.
    lora_scale : float
        LoRA weight scale (only used if lora_path is set).
    prompt : str
        Text prompt conditioning each tile.
    guidance_scale : float
        CFG scale.
    num_inference_steps : int
        Denoising steps.
    tile_size : int
        Tile size in pixels.  1024 is Klein's native resolution.
    tile_overlap : int
        Overlap between adjacent tiles in pixels for feathered blending.
    seed : int
        RNG seed for reproducibility.
    dtype : str
        ``"bfloat16"`` or ``"float16"``.
    device : str
        CUDA device.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        lora_path: Optional[str] = _DEFAULT_LORA,
        lora_scale: float = 1.0,
        prompt: str = "extremely detailed photorealistic baby skin, skin pores, subsurface scattering, 8k texture",
        guidance_scale: float = 4.0,
        num_inference_steps: int = 28,
        tile_size: int = 1024,
        tile_overlap: int = 256,
        seed: int = 42,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        self.model_id = model
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.seed = seed
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self._device = device
        self._pipeline = None

    @property
    def name(self) -> str:
        return "FluxKlein-Tiled"

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from diffusers import Flux2KleinPipeline

        logger.info(f"[FluxKlein] Loading pipeline: {self.model_id}")
        self._pipeline = Flux2KleinPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        )

        if self.lora_path and os.path.exists(self.lora_path):
            logger.info(f"[FluxKlein] Fusing LoRA: {self.lora_path}")
            self._pipeline.load_lora_weights(self.lora_path)
            self._pipeline.fuse_lora(lora_scale=self.lora_scale)

        self._pipeline.to(self._device)
        self._pipeline.set_progress_bar_config(disable=True)
        logger.info("[FluxKlein] Pipeline ready.")

    def _unload_pipeline(self):
        if self._pipeline is not None:
            self._pipeline.to("cpu")
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Single tile inference
    # ------------------------------------------------------------------

    def _refine_tile(self, tile_pil: Image.Image) -> Image.Image:
        """Run Klein on a single tile.  The tile is passed as ``image=``
        (reference conditioning) — Klein encodes it and concatenates with
        the noise latents, so the output is a new image faithful to the
        reference but with details guided by the prompt."""
        w, h = tile_pil.size
        # Klein needs dimensions divisible by 16
        w16, h16 = (w // 16) * 16, (h // 16) * 16
        if (w16, h16) != (w, h):
            tile_pil = tile_pil.resize((w16, h16), Image.LANCZOS)

        generator = torch.Generator(device=self._device).manual_seed(self.seed)

        print(f"Refining tile with guidance scale {self.guidance_scale} and steps {self.num_inference_steps}")
        result = self._pipeline(
            prompt=self.prompt,
            image=tile_pil,
            height=h16,
            width=w16,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        if result.size != (w, h):
            result = result.resize((w, h), Image.LANCZOS)
        return result

    # ------------------------------------------------------------------
    # Tiled UV processing
    # ------------------------------------------------------------------

    def _process_uv_tiled(
        self, texture: torch.Tensor, debug_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Process the full UV texture tile-by-tile with feathered blending."""
        H, W, C = texture.shape
        tiles = _tile_coords(H, W, self.tile_size, self.tile_overlap)
        logger.info(
            f"[FluxKlein] {len(tiles)} tiles ({self.tile_size}px, "
            f"overlap={self.tile_overlap}px) on {H}×{W} texture"
        )

        result = torch.zeros_like(texture)
        weight = torch.zeros(H, W, 1, device=texture.device)

        n_processed = 0
        n_skipped = 0

        for idx, (y0, x0, y1, x1) in enumerate(tiles):
            th, tw = y1 - y0, x1 - x0
            tile_tensor = texture[y0:y1, x0:x1, :]

            if _is_empty(tile_tensor):
                n_skipped += 1
                continue

            tile_np = (tile_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            tile_pil = Image.fromarray(tile_np)

            needs_resize = (th != self.tile_size or tw != self.tile_size)
            if needs_resize:
                tile_pil = tile_pil.resize((self.tile_size, self.tile_size), Image.LANCZOS)

            refined_pil = self._refine_tile(tile_pil)

            if needs_resize:
                refined_pil = refined_pil.resize((tw, th), Image.LANCZOS)

            if debug_dir:
                td = os.path.join(debug_dir, "tiles")
                os.makedirs(td, exist_ok=True)
                tile_pil_dbg = tile_pil if not needs_resize else tile_pil.resize((tw, th), Image.LANCZOS)
                tile_pil_dbg.save(os.path.join(td, f"tile_{idx:03d}_input.png"))
                refined_pil.save(os.path.join(td, f"tile_{idx:03d}_output.png"))

            refined_t = torch.from_numpy(
                np.array(refined_pil).astype(np.float32) / 255.0
            ).to(texture.device)

            fmask = _feather_mask(th, tw, self.tile_overlap, texture.device).unsqueeze(-1)
            result[y0:y1, x0:x1, :] += refined_t * fmask
            weight[y0:y1, x0:x1, :] += fmask
            n_processed += 1

            logger.info(f"[FluxKlein] Tile {idx+1}/{len(tiles)} done.")

        # Normalize by accumulated weights; keep original where no tile covered
        covered = weight.squeeze(-1) > 1e-8
        result[covered] = result[covered] / weight.expand_as(result)[covered]
        result[~covered] = texture[~covered]

        logger.info(
            f"[FluxKlein] Tiled processing complete: "
            f"{n_processed} processed, {n_skipped} skipped (empty)."
        )
        return result

    # ------------------------------------------------------------------
    # UV-space entry point (dispatched by SkinTextureRefiner)
    # ------------------------------------------------------------------

    def refine_uv(
        self,
        texture: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        debug_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """Refine UV texture in-place using tiled Klein inference.

        Parameters
        ----------
        texture : (H, W, 3) float tensor [0, 1]
        mask : (H, W) or (H, W, 1) float tensor, optional
        debug_dir : str, optional

        Returns
        -------
        refined : (H, W, 3) float tensor [0, 1]
        """
        logger.info(
            f"[FluxKlein] UV-space tiled refinement "
            f"(tile={self.tile_size}, overlap={self.tile_overlap}, "
            f"guidance={self.guidance_scale}, steps={self.num_inference_steps})"
        )

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            _save_debug(texture, os.path.join(debug_dir, "klein_input.png"))

        self._load_pipeline()
        refined = self._process_uv_tiled(texture.clamp(0, 1), debug_dir)
        self._unload_pipeline()

        refined = refined.clamp(0, 1)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(texture.device).float()
            refined = mask * refined + (1 - mask) * texture

        if debug_dir:
            _save_debug(refined, os.path.join(debug_dir, "klein_output.png"))
            _save_side_by_side(texture, refined, os.path.join(debug_dir, "klein_comparison.png"))

        logger.info("[FluxKlein] UV-space refinement complete.")
        return refined

    # ------------------------------------------------------------------
    # BaseSkinRefiner fallback (per-view mode)
    # ------------------------------------------------------------------

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Fallback: refine a single rendered view."""
        self._load_pipeline()
        try:
            original_size = image.size
            img = image.resize((self.tile_size, self.tile_size), Image.LANCZOS)
            refined = self._refine_tile(img)
            return refined.resize(original_size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"[FluxKlein] restore() failed: {e}")
            return None

    def to(self, device: str):
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_debug(tensor: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_side_by_side(orig: torch.Tensor, refined: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    o = (orig.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    r = (refined.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    sep = np.ones((o.shape[0], 4, 3), dtype=np.uint8) * 180
    Image.fromarray(np.concatenate([o, sep, r], axis=1)).save(path)
