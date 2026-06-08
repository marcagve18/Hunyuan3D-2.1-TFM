"""sweep_sync.py — Ablation study for FluxKleinSync parameters.

Loads the mesh + Klein once, then sweeps sync_start / sync_end / sync_strength,
saving view 0 for each combo and a final comparison grid.

Usage:
    cd hy3dpaint
    python sweep_sync.py \
        --mesh /path/to/unwrapped_cylindrical.obj \
        --texture /path/to/material_0.png \
        --output_dir /path/to/sweep_output
"""

import os
import sys
import copy
import argparse
import logging
import itertools

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DifferentiableRenderer.MeshRender import MeshRender
from skin_refine.refiners.flux_klein_sync import (
    FluxKleinSyncRefiner,
    _build_voxel_map_normalised,
)
from skin_refine.refiner import _render_view

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SweepSync")

VIEWPOINTS = [
    (  0,   0),
    (  0,  30),
    (  0, -30),
    ( 15,   0),
    (-10,   0),
    (  0,  60),
    (  0, -60),
    ( 15,  30),
]

DISPLAY_VIEW = 0  # which view to put in the grid


def render_views(render, viewpoints, res):
    """Render albedo + position maps for all viewpoints."""
    views = []
    position_maps = []
    for elev, azim in viewpoints:
        rendered, _ = _render_view(render, elev, azim, res, "tex")
        pil_view = Image.fromarray(
            (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        views.append(pil_view)
        pos = render.render_position(elev, azim, resolution=(res, res), return_type="th")
        position_maps.append(pos.squeeze(0) if pos.dim() == 4 else pos)
    return views, position_maps


def run_single_config(
    refiner,
    pipe,
    views,
    position_maps,
    prompt_embeds,
    text_ids,
    negative_prompt_embeds,
    negative_text_ids,
    all_image_latents,
    all_image_ids,
    num_channels,
    voxel_map,
    sync_start,
    sync_end,
    sync_strength,
    w16, h16,
    spatial_h, spatial_w,
    seed=42,
):
    """Run one denoising loop with given sync params. Returns list of refined PIL views."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

    n_views = len(views)
    generators = [
        torch.Generator(device=pipe._execution_device).manual_seed(seed + i)
        for i in range(n_views)
    ]

    all_latents = []
    all_latent_ids = []
    for i in range(n_views):
        lat, lat_ids = pipe.prepare_latents(
            batch_size=1, num_latents_channels=num_channels,
            height=h16, width=w16, dtype=prompt_embeds.dtype,
            device=pipe._execution_device, generator=generators[i],
        )
        all_latents.append(lat)
        all_latent_ids.append(lat_ids)

    num_steps = refiner.num_inference_steps
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = all_latents[0].shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    timesteps, num_steps = retrieve_timesteps(
        pipe.scheduler, num_steps, pipe._execution_device, sigmas=sigmas, mu=mu,
    )

    schedulers = []
    for vi in range(n_views):
        sched = copy.deepcopy(pipe.scheduler)
        sched.set_begin_index(0)
        schedulers.append(sched)

    for step_i, t in enumerate(timesteps):
        frac = step_i / max(num_steps - 1, 1)

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
                noise_pred = neg_pred + refiner.guidance_scale * (noise_pred - neg_pred)

            lat_dtype = lat.dtype
            all_latents[vi] = schedulers[vi].step(noise_pred, t, lat, return_dict=False)[0]
            if all_latents[vi].dtype != lat_dtype:
                all_latents[vi] = all_latents[vi].to(lat_dtype)

        # Sync
        if sync_strength > 0 and sync_start <= frac <= sync_end and voxel_map:
            sync_progress = (frac - sync_start) / max(sync_end - sync_start, 1e-6)
            cur_strength = sync_strength * (1.0 - sync_progress)

            spatial_latents = []
            for vi in range(n_views):
                unpacked = pipe._unpack_latents_with_ids(all_latents[vi], all_latent_ids[vi])
                spatial_latents.append(unpacked.squeeze(0))

            refiner._sync_step(spatial_latents, voxel_map, cur_strength)

            for vi in range(n_views):
                all_latents[vi] = pipe._pack_latents(spatial_latents[vi].unsqueeze(0))

    # Decode
    refined = []
    for vi in range(n_views):
        lat = pipe._unpack_latents_with_ids(all_latents[vi], all_latent_ids[vi])
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(lat.device, lat.dtype)
        lat = lat * bn_std + bn_mean
        lat = pipe._unpatchify_latents(lat)
        img = pipe.vae.decode(lat, return_dict=False)[0]
        img = pipe.image_processor.postprocess(img, output_type="pil")[0]
        img = img.resize(views[vi].size, Image.LANCZOS)
        refined.append(img)

    return refined


def make_grid(images, labels, input_view, ncols=4, cell_size=300):
    """Create a labeled comparison grid with the input view in top-left."""
    all_imgs = [input_view] + images
    all_labels = ["INPUT (no refine)"] + labels
    n = len(all_imgs)
    nrows = (n + ncols - 1) // ncols
    grid_w = ncols * cell_size
    grid_h = nrows * (cell_size + 25)
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(all_imgs, all_labels)):
        row, col = divmod(i, ncols)
        x = col * cell_size
        y = row * (cell_size + 25)
        resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(resized, (x, y + 20))
        draw.text((x + 5, y + 2), label, fill=(255, 255, 255), font=font)

    return grid


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to unwrapped OBJ mesh")
    parser.add_argument("--texture", required=True, help="Path to baked UV texture PNG (e.g. texture_input.png from skin_refiner_debug)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--prompt", default="Enhance sharpness and detail, preserve structure. Realistic baby skin texture.")
    parser.add_argument("--guidance", type=float, default=1.4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--res", type=int, default=1024, help="Render/Klein resolution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Parameter grid ---
    sync_starts   = [0.0, 0.05, 0.1, 0.2]
    sync_ends     = [0.3, 0.5, 0.7]
    sync_strengths = [0.1, 0.2, 0.3]
    # + baseline with strength=0 (no sync)
    configs = [{"start": 0, "end": 1, "strength": 0.0, "label": "no_sync"}]
    for start, end, strength in itertools.product(sync_starts, sync_ends, sync_strengths):
        if start >= end:
            continue
        configs.append({
            "start": start, "end": end, "strength": strength,
            "label": f"s{start}_e{end}_w{strength}",
        })
    logger.info(f"Sweep: {len(configs)} configurations")

    # --- Load mesh + texture ---
    logger.info("Loading mesh...")
    render = MeshRender(
        default_resolution=args.res * 2,
        texture_size=1024 * 4,
        bake_mode="back_sample",
        raster_mode="cr",
    )
    mesh = trimesh.load(args.mesh)
    render.load_mesh(mesh=mesh)

    tex_pil = Image.open(args.texture).convert("RGB")
    tex_np = np.array(tex_pil).astype(np.float32) / 255.0
    tex_tensor = torch.from_numpy(tex_np).to("cuda")
    render.set_texture(tex_tensor, force_set=True)

    # --- Render views ---
    logger.info("Rendering views...")
    res = args.res
    views, position_maps = render_views(render, VIEWPOINTS, res)
    for i, v in enumerate(views):
        v.save(os.path.join(args.output_dir, f"input_view_{i:02d}.png"))

    # --- Load Klein ---
    logger.info("Loading Klein pipeline...")
    refiner = FluxKleinSyncRefiner(
        model=args.model,
        prompt=args.prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        width=res, height=res,
        seed=args.seed,
    )
    refiner._load_pipeline()
    pipe = refiner._pipeline
    pipe._guidance_scale = args.guidance
    pipe._attention_kwargs = None

    w16 = (res // 16) * 16
    h16 = (res // 16) * 16
    latent_h = 2 * (h16 // (pipe.vae_scale_factor * 2))
    latent_w = 2 * (w16 // (pipe.vae_scale_factor * 2))
    spatial_h = latent_h // 2
    spatial_w = latent_w // 2

    # --- Prepare shared data (once) ---
    logger.info("Building voxel map...")
    voxel_map = _build_voxel_map_normalised(
        position_maps, 64, spatial_h, spatial_w,
    )
    logger.info(f"Voxel map: {len(voxel_map)} shared voxels")

    logger.info("Encoding prompt...")
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device,
        num_images_per_prompt=1, max_sequence_length=512,
    )
    negative_prompt_embeds, negative_text_ids = None, None
    if pipe.do_classifier_free_guidance:
        negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
            prompt="", device=pipe._execution_device,
            num_images_per_prompt=1, max_sequence_length=512,
        )

    logger.info("Encoding image latents...")
    num_channels = pipe.transformer.config.in_channels // 4
    n_views = len(views)
    all_image_latents = []
    all_image_ids = []
    generators = [
        torch.Generator(device=pipe._execution_device).manual_seed(args.seed + i)
        for i in range(n_views)
    ]
    mult = pipe.vae_scale_factor * 2
    for i, view_pil in enumerate(views):
        img = view_pil.convert("RGB").resize((w16, h16), Image.LANCZOS)
        pipe.image_processor.check_image_input(img)
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

    # --- Sweep ---
    n_views = len(views)
    result_images_per_view = [[] for _ in range(n_views)]
    result_labels = []
    for ci, cfg in enumerate(configs):
        label = cfg["label"]
        logger.info(f"[{ci+1}/{len(configs)}] Running: {label}")

        refined = run_single_config(
            refiner=refiner,
            pipe=pipe,
            views=views,
            position_maps=position_maps,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            all_image_latents=all_image_latents,
            all_image_ids=all_image_ids,
            num_channels=num_channels,
            voxel_map=voxel_map,
            sync_start=cfg["start"],
            sync_end=cfg["end"],
            sync_strength=cfg["strength"],
            w16=w16, h16=h16,
            spatial_h=spatial_h, spatial_w=spatial_w,
            seed=args.seed,
        )

        for vi in range(n_views):
            out_path = os.path.join(args.output_dir, f"{label}_view{vi:02d}.png")
            refined[vi].save(out_path)
            result_images_per_view[vi].append(refined[vi])
        result_labels.append(label)
        logger.info(f"  Saved {n_views} views for: {label}")

    # --- Per-view grids ---
    logger.info("Generating per-view comparison grids...")
    for vi in range(n_views):
        grid = make_grid(result_images_per_view[vi], result_labels, views[vi])
        grid_path = os.path.join(args.output_dir, f"comparison_grid_view{vi:02d}.png")
        grid.save(grid_path)
        logger.info(f"Grid saved: {grid_path}")


if __name__ == "__main__":
    main()
