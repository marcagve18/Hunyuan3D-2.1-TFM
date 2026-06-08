"""skin_refine/refiners/flux_klein_sync.py — SyncDiffusion-style Klein multi-view refiner.
=============================================================================================
Processes all views jointly through Klein's denoising loop, synchronising
latents at each step so that overlapping 3-D regions converge to the same
texture.  No retraining — the sync is a post-hoc latent blend injected into
the standard Klein scheduler loop.

Algorithm
---------
1.  Render N views (albedo + position maps).
2.  VAE-encode each view → per-view latents.
3.  Build a *voxel correspondence map*: discretise 3-D positions into a
    fixed voxel grid; for every voxel record which (view, h, w) pixels map
    to it.  Down-sample to latent resolution (÷8 for Flux VAE).
4.  Custom denoising loop:
      for each timestep t:
          a.  UNet forward per view → noise_pred_v
          b.  scheduler.step per view → latents_v
          c.  **sync**: for each voxel that is visible in >1 view, replace
              latents_v[h,w] with the cosine-weighted mean across views.
              Strength decays linearly from `sync_start` to `sync_end`.
5.  VAE-decode → refined views → back-project + bake.

The refiner exposes ``refine_multiview()`` so the orchestrator can bypass
the default per-view ``restore()`` loop.
"""

import logging
import os
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


# ---------------------------------------------------------------------------
# Voxel correspondence builder
# ---------------------------------------------------------------------------

def _build_voxel_map(
    position_maps: List[torch.Tensor],
    voxel_resolution: int,
    latent_h: int,
    latent_w: int,
) -> Dict[int, List[Tuple[int, int, int]]]:
    """Build a mapping from voxel index → list of (view_idx, lat_h, lat_w).

    Parameters
    ----------
    position_maps : list of (H, W, 3) tensors in world-space [0, 1]-ish.
    voxel_resolution : grid cells per axis.
    latent_h, latent_w : spatial dimensions of the latent tensor.

    Returns
    -------
    voxel_to_pixels : dict  voxel_flat_idx → [(view, lh, lw), …]
        Only voxels visible in ≥2 views are included.
    """
    voxel_to_pixels: Dict[int, List[Tuple[int, int, int]]] = {}
    V = voxel_resolution

    for vi, pmap in enumerate(position_maps):
        # Downsample position map to latent resolution
        # pmap: (H, W, 3)
        pmap_down = F.interpolate(
            pmap.permute(2, 0, 1).unsqueeze(0),
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)  # (latent_h, latent_w, 3)

        # Visibility mask: non-zero positions
        vis = pmap_down.abs().sum(-1) > 1e-4  # (latent_h, latent_w)

        # Normalise positions to [0, 1] per-axis (use global bounds across all maps)
        pos = pmap_down[vis]  # (N, 3)
        if pos.numel() == 0:
            continue

        coords = pos.cpu().numpy()
        hs, ws = torch.where(vis)
        hs, ws = hs.cpu().numpy(), ws.cpu().numpy()

        for i in range(len(coords)):
            # Quantise to voxel grid
            vx = int(np.clip(coords[i, 0] * V, 0, V - 1))
            vy = int(np.clip(coords[i, 1] * V, 0, V - 1))
            vz = int(np.clip(coords[i, 2] * V, 0, V - 1))
            flat = vx * V * V + vy * V + vz
            voxel_to_pixels.setdefault(flat, []).append((vi, int(hs[i]), int(ws[i])))

    # Keep only multi-view voxels
    return {k: v for k, v in voxel_to_pixels.items() if len(set(e[0] for e in v)) > 1}


def _build_voxel_map_normalised(
    position_maps: List[torch.Tensor],
    voxel_resolution: int,
    latent_h: int,
    latent_w: int,
) -> Dict[int, List[Tuple[int, int, int]]]:
    """Same as _build_voxel_map but normalises coordinates globally first."""
    V = voxel_resolution

    # Collect all visible positions to compute global bounds
    all_positions = []
    downsampled = []
    vis_masks = []
    for pmap in position_maps:
        pmap_down = F.interpolate(
            pmap.permute(2, 0, 1).unsqueeze(0),
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        vis = pmap_down.abs().sum(-1) > 1e-4
        downsampled.append(pmap_down)
        vis_masks.append(vis)
        all_positions.append(pmap_down[vis])

    if not all_positions or sum(p.numel() for p in all_positions) == 0:
        return {}

    all_pos = torch.cat(all_positions, dim=0)
    pos_min = all_pos.min(0).values
    pos_max = all_pos.max(0).values
    pos_range = (pos_max - pos_min).clamp(min=1e-6)

    voxel_to_pixels: Dict[int, List[Tuple[int, int, int]]] = {}
    for vi, (pmap_down, vis) in enumerate(zip(downsampled, vis_masks)):
        normed = (pmap_down - pos_min) / pos_range  # [0, 1]
        hs, ws = torch.where(vis)
        coords = normed[vis].cpu().numpy()
        hs, ws = hs.cpu().numpy(), ws.cpu().numpy()

        for i in range(len(coords)):
            vx = int(np.clip(coords[i, 0] * (V - 1), 0, V - 1))
            vy = int(np.clip(coords[i, 1] * (V - 1), 0, V - 1))
            vz = int(np.clip(coords[i, 2] * (V - 1), 0, V - 1))
            flat = vx * V * V + vy * V + vz
            voxel_to_pixels.setdefault(flat, []).append((vi, int(hs[i]), int(ws[i])))

    return {k: v for k, v in voxel_to_pixels.items() if len(set(e[0] for e in v)) > 1}


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class FluxKleinSyncRefiner(BaseSkinRefiner):
    """Synchronized multi-view Klein refiner (SyncDiffusion-style).

    Parameters
    ----------
    model : str
        HuggingFace model ID for Klein 4B.
    lora_path, lora_scale : LoRA config (optional).
    prompt : str
        Text prompt guiding the edit.
    guidance_scale : float
    num_inference_steps : int
    width, height : int
        Resolution Klein runs at per view.
    seed : int
    dtype : str  ``"bfloat16"`` | ``"float16"``
    device : str
    sync_strength : float
        Peak blending weight toward the cross-view mean (0 = no sync, 1 = full).
    sync_start : float
        Fraction of denoising at which sync begins (0 = from the start).
    sync_end : float
        Fraction of denoising at which sync ends.
    voxel_resolution : int
        Grid cells per axis for the 3-D correspondence voxel map.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        prompt: str = "Enhance sharpness and detail, preserve structure",
        guidance_scale: float = 1.4,
        num_inference_steps: int = 40,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        dtype: str = "bfloat16",
        device: str = "cuda",
        sync_strength: float = 0.6,
        sync_start: float = 0.0,
        sync_end: float = 0.8,
        voxel_resolution: int = 64,
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
        self.sync_strength = sync_strength
        self.sync_start = sync_start
        self.sync_end = sync_end
        self.voxel_resolution = voxel_resolution
        self._pipeline = None
        self.debug_dir = None

    @property
    def name(self) -> str:
        return "FluxKlein-Sync"

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from diffusers import Flux2KleinPipeline

        logger.info(f"[FluxKleinSync] Loading pipeline: {self.model_id}")
        self._pipeline = Flux2KleinPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        )

        if self.lora_path and os.path.exists(self.lora_path):
            logger.info(f"[FluxKleinSync] Fusing LoRA: {self.lora_path}")
            self._pipeline.load_lora_weights(self.lora_path)
            self._pipeline.fuse_lora(lora_scale=self.lora_scale)

        self._pipeline.to(self._device)
        logger.info("[FluxKleinSync] Pipeline ready.")

    # ------------------------------------------------------------------
    # Per-view restore (fallback — not the intended path)
    # ------------------------------------------------------------------

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Fallback: process a single view without sync."""
        self._load_pipeline()
        try:
            w16 = (self.width // 16) * 16
            h16 = (self.height // 16) * 16
            img = image.convert("RGB").resize((w16, h16), Image.LANCZOS)
            gen = torch.Generator(device=self._device).manual_seed(self.seed)
            result = self._pipeline(
                prompt=self.prompt, image=img, height=h16, width=w16,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=gen,
            ).images[0]
            return result.resize(image.size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"[FluxKleinSync] restore() failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _save_latent_snapshot(self, pipe, packed_latent, latent_ids, path):
        """Decode a single view's latent and save as image for debugging."""
        try:
            lat = pipe._unpack_latents_with_ids(packed_latent, latent_ids)
            bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
            bn_std = torch.sqrt(
                pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
            ).to(lat.device, lat.dtype)
            lat = lat * bn_std + bn_mean
            lat = pipe._unpatchify_latents(lat)
            img = pipe.vae.decode(lat, return_dict=False)[0]
            img = pipe.image_processor.postprocess(img, output_type="pil")[0]
            img.save(path)
        except Exception as e:
            logger.warning(f"[FluxKleinSync] Snapshot failed: {e}")

    # ------------------------------------------------------------------
    # Synchronized multi-view refinement
    # ------------------------------------------------------------------

    def _sync_step(
        self,
        latents_list: List[torch.Tensor],
        voxel_map: Dict[int, List[Tuple[int, int, int]]],
        strength: float,
    ):
        """In-place latent sync across views via voxel correspondence.

        latents_list : list of (C, H, W) tensors — one per view.
        """
        if strength < 1e-4 or not voxel_map:
            return

        n_synced = 0
        for _voxel_idx, pixels in voxel_map.items():
            # Gather latent vectors at corresponding positions
            vals = []
            for (vi, lh, lw) in pixels:
                vals.append(latents_list[vi][:, lh, lw])  # (C,)
            mean_val = torch.stack(vals).mean(0)

            # Blend toward mean
            for idx, (vi, lh, lw) in enumerate(pixels):
                latents_list[vi][:, lh, lw] = (
                    (1 - strength) * latents_list[vi][:, lh, lw]
                    + strength * mean_val
                )
            n_synced += 1

        return n_synced

    @torch.no_grad()
    def refine_multiview(
        self,
        views: List[Image.Image],
        position_maps: List[torch.Tensor],
        render,
        normal_maps: Optional[List[Image.Image]] = None,
        debug_dir: Optional[str] = None,
    ) -> List[Image.Image]:
        """Run synchronized Klein denoising on all views jointly.

        Parameters
        ----------
        views : list of PIL images (rendered albedo per viewpoint).
        position_maps : list of (H, W, 3) tensors (world-space positions).
        render : MeshRender (needed for back_project later, not used here).
        normal_maps : list of PIL images (rendered normals per viewpoint, optional).
        debug_dir : optional path for intermediate images.

        Returns
        -------
        refined_views : list of PIL images, same order as input.
        """
        self._load_pipeline()
        pipe = self._pipeline
        pipe._guidance_scale = self.guidance_scale
        pipe._attention_kwargs = None
        n_views = len(views)

        w16 = (self.width // 16) * 16
        h16 = (self.height // 16) * 16

        # --- Latent spatial dims (Klein uses 2× patchify on top of 8× VAE) ---
        latent_h = 2 * (h16 // (pipe.vae_scale_factor * 2))
        latent_w = 2 * (w16 // (pipe.vae_scale_factor * 2))
        # After patchify: spatial dims are latent_h//2, latent_w//2
        # but sync operates on the packed sequence. For voxel map we use
        # the pre-pack spatial dims.
        spatial_h = latent_h // 2
        spatial_w = latent_w // 2

        logger.info(
            f"[FluxKleinSync] {n_views} views @ {w16}×{h16}, "
            f"latent {spatial_h}×{spatial_w}, "
            f"sync_strength={self.sync_strength}, "
            f"voxel_res={self.voxel_resolution}"
        )

        # 1. Build voxel correspondence map
        voxel_map = _build_voxel_map_normalised(
            position_maps, self.voxel_resolution, spatial_h, spatial_w,
        )
        logger.info(f"[FluxKleinSync] Voxel map: {len(voxel_map)} shared voxels")

        # Convert position maps to PIL for use as Klein reference images
        position_pils = []
        for pmap in position_maps:
            p = pmap.cpu()
            if p.dim() == 4:
                p = p.squeeze(0)
            # Normalise to [0, 1] for visualisation
            p_min = p.min()
            p_range = (p.max() - p_min).clamp(min=1e-6)
            p_norm = ((p - p_min) / p_range * 255).clamp(0, 255).byte().numpy()
            position_pils.append(Image.fromarray(p_norm))

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            for i, v in enumerate(views):
                v.save(os.path.join(debug_dir, f"input_view_{i:02d}.png"))
            if normal_maps:
                for i, n in enumerate(normal_maps):
                    n.save(os.path.join(debug_dir, f"input_normal_{i:02d}.png"))
            for i, p in enumerate(position_pils):
                p.save(os.path.join(debug_dir, f"input_position_{i:02d}.png"))

        logger.info("[FluxKleinSync] Klein reference: albedo only. "
                    "Normal + position used for voxel sync correspondence.")

        # 2. Encode prompt (shared across views)
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=self.prompt, device=pipe._execution_device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
        negative_prompt_embeds, negative_text_ids = None, None
        if pipe.do_classifier_free_guidance:
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt="", device=pipe._execution_device,
                num_images_per_prompt=1, max_sequence_length=512,
            )

        # 3. Prepare per-view latents and image conditioning
        num_channels = pipe.transformer.config.in_channels // 4
        generators = [
            torch.Generator(device=self._device).manual_seed(self.seed + i)
            for i in range(n_views)
        ]

        all_latents = []       # packed: (1, seq, C)
        all_latent_ids = []
        all_image_latents = [] # packed: (1, seq, C)
        all_image_ids = []

        for i, view_pil in enumerate(views):
            img = view_pil.convert("RGB").resize((w16, h16), Image.LANCZOS)

            # Noise latents
            lat, lat_ids = pipe.prepare_latents(
                batch_size=1, num_latents_channels=num_channels,
                height=h16, width=w16, dtype=prompt_embeds.dtype,
                device=pipe._execution_device, generator=generators[i],
            )
            all_latents.append(lat)
            all_latent_ids.append(lat_ids)

            # Klein reference: only albedo (normal/position are used for sync, not as references)
            mult = pipe.vae_scale_factor * 2
            pipe.image_processor.check_image_input(img)
            img_w, img_h = img.size
            if img_w * img_h > 1024 * 1024:
                img = pipe.image_processor._resize_to_target_area(img, 1024 * 1024)
                img_w, img_h = img.size
            img_w = (img_w // mult) * mult
            img_h = (img_h // mult) * mult
            img_tensor = pipe.image_processor.preprocess(img, height=img_h, width=img_w, resize_mode="crop")

            im_lat, im_ids = pipe.prepare_image_latents(
                images=[img_tensor], batch_size=1,
                generator=generators[i], device=pipe._execution_device,
                dtype=pipe.vae.dtype,
            )
            all_image_latents.append(im_lat)
            all_image_ids.append(im_ids)

        # 4. Prepare timesteps
        import numpy as np_local
        sigmas = np_local.linspace(1.0, 1.0 / self.num_inference_steps, self.num_inference_steps)
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = all_latents[0].shape[1]

        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.num_inference_steps)
        timesteps, num_steps = retrieve_timesteps(
            pipe.scheduler, self.num_inference_steps,
            pipe._execution_device, sigmas=sigmas, mu=mu,
        )
        import copy
        schedulers = []
        for vi in range(n_views):
            sched = copy.deepcopy(pipe.scheduler)
            sched.set_begin_index(0)
            schedulers.append(sched)

        # Debug: which steps to snapshot (first, 25%, 50%, 75%, last)
        _dbg_steps = set()
        if debug_dir:
            for pct in [0, 0.25, 0.5, 0.75, 1.0]:
                _dbg_steps.add(min(int(pct * (num_steps - 1)), num_steps - 1))
            _dbg_snap_dir = os.path.join(debug_dir, "denoising_snapshots")
            os.makedirs(_dbg_snap_dir, exist_ok=True)

        # 5. Denoising loop with sync
        for step_i, t in enumerate(timesteps):
            frac = step_i / max(num_steps - 1, 1)

            # --- Per-view UNet forward ---
            for vi in range(n_views):
                lat = all_latents[vi]
                timestep = t.expand(lat.shape[0]).to(lat.dtype)

                lat_input = lat.to(pipe.transformer.dtype)
                lat_ids = all_latent_ids[vi]
                if all_image_latents[vi] is not None:
                    lat_input = torch.cat([lat, all_image_latents[vi]], dim=1).to(pipe.transformer.dtype)
                    lat_ids = torch.cat([all_latent_ids[vi], all_image_ids[vi]], dim=1)

                noise_pred = pipe.transformer(
                    hidden_states=lat_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=lat_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :lat.size(1)]

                if pipe.do_classifier_free_guidance:
                    neg_pred = pipe.transformer(
                        hidden_states=lat_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=lat_ids,
                        return_dict=False,
                    )[0]
                    neg_pred = neg_pred[:, :lat.size(1)]
                    noise_pred = neg_pred + self.guidance_scale * (noise_pred - neg_pred)

                lat_dtype = lat.dtype
                all_latents[vi] = schedulers[vi].step(noise_pred, t, lat, return_dict=False)[0]
                if all_latents[vi].dtype != lat_dtype:
                    all_latents[vi] = all_latents[vi].to(lat_dtype)

            # Debug: decode view 0 before sync
            if debug_dir and step_i in _dbg_steps:
                self._save_latent_snapshot(
                    pipe, all_latents[0], all_latent_ids[0],
                    os.path.join(_dbg_snap_dir, f"step{step_i:03d}_v0_before_sync.png"),
                )

            # --- Sync step ---
            if self.sync_start <= frac <= self.sync_end and self.sync_strength > 0:
                # Linear decay of strength
                sync_progress = (frac - self.sync_start) / max(self.sync_end - self.sync_start, 1e-6)
                cur_strength = self.sync_strength * (1.0 - sync_progress)

                # Unpack latents to spatial form for sync
                spatial_latents = []
                for vi in range(n_views):
                    unpacked = pipe._unpack_latents_with_ids(
                        all_latents[vi], all_latent_ids[vi],
                    )  # (1, C, H, W)
                    spatial_latents.append(unpacked.squeeze(0))  # (C, H, W)

                n_synced = self._sync_step(spatial_latents, voxel_map, cur_strength)

                # Re-pack
                for vi in range(n_views):
                    repacked = pipe._pack_latents(spatial_latents[vi].unsqueeze(0))
                    all_latents[vi] = repacked

                # Debug: decode view 0 after sync
                if debug_dir and step_i in _dbg_steps:
                    self._save_latent_snapshot(
                        pipe, all_latents[0], all_latent_ids[0],
                        os.path.join(_dbg_snap_dir, f"step{step_i:03d}_v0_after_sync.png"),
                    )

                if step_i % 10 == 0:
                    logger.info(
                        f"[FluxKleinSync] Step {step_i}/{num_steps}: "
                        f"synced {n_synced} voxels, strength={cur_strength:.3f}"
                    )

        # 6. Decode
        refined_views = []
        for vi in range(n_views):
            lat = pipe._unpack_latents_with_ids(all_latents[vi], all_latent_ids[vi])
            # BatchNorm denormalisation
            bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
            bn_std = torch.sqrt(
                pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
            ).to(lat.device, lat.dtype)
            lat = lat * bn_std + bn_mean
            lat = pipe._unpatchify_latents(lat)

            img = pipe.vae.decode(lat, return_dict=False)[0]
            img = pipe.image_processor.postprocess(img, output_type="pil")[0]

            original_size = views[vi].size
            img = img.resize(original_size, Image.LANCZOS)
            refined_views.append(img)

            if debug_dir:
                img.save(os.path.join(debug_dir, f"refined_view_{vi:02d}.png"))

        logger.info(f"[FluxKleinSync] Refined {n_views} views with cross-view sync.")
        return refined_views

    def to(self, device: str):
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self
