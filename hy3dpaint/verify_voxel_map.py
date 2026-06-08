"""verify_voxel_map.py — Visualize voxel correspondence across views.

For each shared voxel, draws matching pixels in the same random color
on every view where the voxel is visible. Produces:
  1. Per-view overlay images (albedo + colored correspondence dots)
  2. A pair-wise view grid showing shared correspondences between every two views
  3. Stats: how many voxels, coverage per view, histogram of voxel occupancy

Usage:
    cd hy3dpaint
    python verify_voxel_map.py \
        --mesh /path/to/unwrapped_cylindrical.obj \
        --texture /path/to/texture_input.png \
        --output_dir /path/to/voxel_debug
"""

import os
import sys
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DifferentiableRenderer.MeshRender import MeshRender
from skin_refine.refiners.flux_klein_sync import _build_voxel_map_normalised
from skin_refine.refiner import _render_view

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


def render_all(render, viewpoints, res):
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


def random_color(seed_val):
    rng = random.Random(seed_val)
    return (rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--voxel_res", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load mesh ---
    print("Loading mesh...")
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
    print("Rendering views...")
    res = args.res
    views, position_maps = render_all(render, VIEWPOINTS, res)
    for i, v in enumerate(views):
        v.save(os.path.join(args.output_dir, f"view_{i:02d}.png"))

    # --- Compute latent spatial dims (matching Klein's pipeline) ---
    # Klein: 8x VAE + 2x patchify → spatial latent = res / 16
    spatial_h = res // 16
    spatial_w = res // 16

    # --- Build voxel map ---
    print(f"Building voxel map (voxel_res={args.voxel_res}, latent={spatial_h}x{spatial_w})...")
    voxel_map = _build_voxel_map_normalised(
        position_maps, args.voxel_res, spatial_h, spatial_w,
    )

    n_views = len(views)
    scale_h = res / spatial_h
    scale_w = res / spatial_w

    # --- Stats ---
    print(f"\n{'='*60}")
    print(f"VOXEL MAP STATISTICS")
    print(f"{'='*60}")
    print(f"Total shared voxels (visible in 2+ views): {len(voxel_map)}")

    views_per_voxel = [len(set(vi for vi, _, _ in pixels)) for pixels in voxel_map.values()]
    pixels_per_voxel = [len(pixels) for pixels in voxel_map.values()]

    if views_per_voxel:
        print(f"Views per voxel:  min={min(views_per_voxel)}, max={max(views_per_voxel)}, "
              f"mean={np.mean(views_per_voxel):.1f}")
        print(f"Pixels per voxel: min={min(pixels_per_voxel)}, max={max(pixels_per_voxel)}, "
              f"mean={np.mean(pixels_per_voxel):.1f}")

    # Per-view coverage
    pixels_per_view = defaultdict(set)
    for voxel_idx, pixels in voxel_map.items():
        for vi, lh, lw in pixels:
            pixels_per_view[vi].add((lh, lw))

    print(f"\nPer-view coverage (latent pixels participating in sync):")
    total_latent_pixels = spatial_h * spatial_w
    for vi in range(n_views):
        elev, azim = VIEWPOINTS[vi]
        count = len(pixels_per_view[vi])
        pct = 100 * count / total_latent_pixels
        print(f"  View {vi} (elev={elev:+3d}, azim={azim:+3d}): "
              f"{count}/{total_latent_pixels} pixels ({pct:.1f}%)")

    # View-pair overlap
    print(f"\nPairwise view overlap (shared voxels between each pair):")
    pair_counts = defaultdict(int)
    for voxel_idx, pixels in voxel_map.items():
        vids = sorted(set(vi for vi, _, _ in pixels))
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                pair_counts[(vids[i], vids[j])] += 1

    for vi in range(n_views):
        for vj in range(vi + 1, n_views):
            count = pair_counts.get((vi, vj), 0)
            print(f"  View {vi} <-> View {vj}: {count} shared voxels")

    # --- Visualization 1: Per-view overlay ---
    print("\nGenerating per-view overlays...")
    dot_radius = max(2, res // 200)

    overlay_views = [v.copy().convert("RGBA") for v in views]
    dot_layers = [Image.new("RGBA", (res, res), (0, 0, 0, 0)) for _ in range(n_views)]
    dot_draws = [ImageDraw.Draw(dl) for dl in dot_layers]

    for voxel_idx, pixels in voxel_map.items():
        color = random_color(voxel_idx)
        color_rgba = color + (180,)
        for vi, lh, lw in pixels:
            cx = int((lw + 0.5) * scale_w)
            cy = int((lh + 0.5) * scale_h)
            dot_draws[vi].ellipse(
                [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                fill=color_rgba,
            )

    for vi in range(n_views):
        overlay = Image.alpha_composite(overlay_views[vi], dot_layers[vi])
        overlay.convert("RGB").save(os.path.join(args.output_dir, f"overlay_view_{vi:02d}.png"))

    # --- Visualization 2: Pairwise correspondence ---
    print("Generating pairwise correspondence images...")
    pair_dir = os.path.join(args.output_dir, "pairs")
    os.makedirs(pair_dir, exist_ok=True)

    cell = min(512, res)
    for vi in range(n_views):
        for vj in range(vi + 1, n_views):
            if pair_counts.get((vi, vj), 0) == 0:
                continue

            img_left = views[vi].resize((cell, cell), Image.LANCZOS).convert("RGBA")
            img_right = views[vj].resize((cell, cell), Image.LANCZOS).convert("RGBA")
            pair_img = Image.new("RGBA", (cell * 2 + 20, cell), (40, 40, 40, 255))
            pair_img.paste(img_left, (0, 0))
            pair_img.paste(img_right, (cell + 20, 0))
            draw = ImageDraw.Draw(pair_img)

            pair_scale_h = cell / spatial_h
            pair_scale_w = cell / spatial_w

            n_lines = 0
            max_lines = 200
            relevant_voxels = [
                (voxel_idx, pixels) for voxel_idx, pixels in voxel_map.items()
                if any(v == vi for v, _, _ in pixels) and any(v == vj for v, _, _ in pixels)
            ]
            step = max(1, len(relevant_voxels) // max_lines)

            for k, (voxel_idx, pixels) in enumerate(relevant_voxels):
                if k % step != 0:
                    continue
                color = random_color(voxel_idx)
                pts_i = [(lh, lw) for v, lh, lw in pixels if v == vi]
                pts_j = [(lh, lw) for v, lh, lw in pixels if v == vj]
                if not pts_i or not pts_j:
                    continue

                lh_i, lw_i = pts_i[0]
                lh_j, lw_j = pts_j[0]
                x1 = int((lw_i + 0.5) * pair_scale_w)
                y1 = int((lh_i + 0.5) * pair_scale_h)
                x2 = int((lw_j + 0.5) * pair_scale_w) + cell + 20
                y2 = int((lh_j + 0.5) * pair_scale_h)

                draw.line([(x1, y1), (x2, y2)], fill=color + (120,), width=1)
                r = 3
                draw.ellipse([x1-r, y1-r, x1+r, y1+r], fill=color + (220,))
                draw.ellipse([x2-r, y2-r, x2+r, y2+r], fill=color + (220,))
                n_lines += 1

            pair_img.convert("RGB").save(
                os.path.join(pair_dir, f"pair_{vi}_{vj}.png")
            )

    # --- Visualization 3: Voxel occupancy heatmap per view ---
    print("Generating voxel density heatmaps...")
    for vi in range(n_views):
        heatmap = np.zeros((spatial_h, spatial_w), dtype=np.float32)
        for voxel_idx, pixels in voxel_map.items():
            n_other_views = len(set(v for v, _, _ in pixels if v != vi))
            for v, lh, lw in pixels:
                if v == vi:
                    heatmap[lh, lw] += n_other_views

        if heatmap.max() > 0:
            heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)

        heatmap_pil = Image.fromarray(heatmap_norm, mode="L").resize(
            (res, res), Image.NEAREST
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colored = cm.inferno(heatmap_norm / 255.0)
        colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_color = Image.fromarray(colored_uint8).resize((res, res), Image.NEAREST)

        blended = Image.blend(
            views[vi].convert("RGB").resize((res, res)),
            heatmap_color.convert("RGB"),
            alpha=0.4,
        )
        blended.save(os.path.join(args.output_dir, f"heatmap_view_{vi:02d}.png"))

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
