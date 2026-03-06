import os
import torch
import trimesh
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image
import glob

from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
# from utils.image_super_utils import imageSuperNet # Commented out ESRGAN
from utils.uvwrap_utils import mesh_uv_wrap
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb

import warnings
warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity(50)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Hunyuan3DPaint")

class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"
        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"
        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)

class Hunyuan3DPaintPipeline:
    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        # Models are now loaded lazily inside __call__

    def load_models(self):
        if "multiview_model" in self.models:
            return
        logger.info("Loading AI Models...")
        torch.cuda.empty_cache()
        # self.models["super_model"] = imageSuperNet(self.config) # ESRGAN disabled
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        logger.info("Models Loaded.")

    def _ensure_dir(self, p: str):
        os.makedirs(p, exist_ok=True)

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8: return arr
        a = arr
        if np.issubdtype(a.dtype, np.floating):
            a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
            mn, mx = float(a.min()), float(a.max())
            if 0.0 <= mn and mx <= 1.0: a = a * 255.0
            else: a = (a - mn) / (mx - mn + 1e-8) * 255.0
        return np.clip(a, 0, 255).astype(np.uint8)

    def _save_np_image(self, arr: np.ndarray, path: str):
        self._ensure_dir(os.path.dirname(path))
        if isinstance(arr, torch.Tensor): arr = arr.detach().cpu().numpy()
        if arr.ndim == 4: arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4): arr = np.transpose(arr[:3], (1, 2, 0))
        if arr.ndim == 2: im = Image.fromarray(self._to_uint8(arr), mode="L")
        else: im = Image.fromarray(self._to_uint8(arr[:, :, :3]), mode="RGB")
        im.save(path)

    def _save_pil(self, img: Image.Image, path: str):
        self._ensure_dir(os.path.dirname(path))
        img.save(path)

    def _save_list_as_images(self, imgs, out_dir: str, prefix: str):
        self._ensure_dir(out_dir)
        for i, im in enumerate(imgs):
            out_path = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
            if isinstance(im, Image.Image): self._save_pil(im, out_path)
            else: self._save_np_image(im, out_path)

    def _load_images_from_dir(self, img_dir: str, pattern: str = "*.png"):
        paths = sorted(glob.glob(os.path.join(img_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"No images found in {img_dir} with pattern {pattern}")
        return [Image.open(p).convert("RGB") for p in paths]

    def _save_uv_visualization(self, mesh, path):
        uvs = mesh.visual.uv
        faces = mesh.faces
        plt.figure(figsize=(8, 8))
        plt.triplot(uvs[:, 0], uvs[:, 1], faces, lw=0.2, color="blue", alpha=0.5)
        plt.xlim(0, 1); plt.ylim(0, 1)
        self._ensure_dir(os.path.dirname(path))
        plt.savefig(path)
        plt.close()

    def apply_cylindrical_unwrap(self, mesh):
        vertices = mesh.vertices - mesh.bounding_box.centroid
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        theta = np.arctan2(x, z)
        u = (theta + np.pi) / (2 * np.pi)
        v = (y - y.min()) / (y.max() - y.min())
        uvs = np.column_stack((u, v))
        uvs[:, 0] = np.clip(uvs[:, 0], 0.005, 0.995)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        return mesh

    def apply_projective_unwrap(self, mesh):
        vertices = mesh.vertices - mesh.bounding_box.centroid
        u = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
        v = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
        uvs = np.column_stack((u * 0.9 + 0.05, v * 0.9 + 0.05))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        return mesh

    @torch.no_grad()
    def __call__(
        self,
        mesh_path=None,
        image_path=None,
        output_mesh_path=None,
        use_remesh=False,
        save_glb=True,
        unwrap_mode='cylindrical',
        use_cache=True,
        manual_albedo_dir=None # NEW: Specify manual folder for enhanced images
    ):
        input_dir = os.path.dirname(mesh_path)
        if output_mesh_path is None:
            output_mesh_path = os.path.join(input_dir, f"textured_{unwrap_mode}.obj")

        inter_dir = os.path.join(os.path.dirname(output_mesh_path), "intermediates")
        dbg_dir = os.path.join(inter_dir, "debug_images")
        
        # 1 & 2) Mesh & UV Caching
        unwrapped_obj = os.path.join(inter_dir, f"unwrapped_{unwrap_mode}.obj")
        if use_cache and os.path.exists(unwrapped_obj):
            logger.info(f"Loading cached unwrapped mesh: {unwrapped_obj}")
            mesh = trimesh.load(unwrapped_obj)
        else:
            if use_remesh:
                processed_mesh_path = os.path.join(inter_dir, "step1_remeshed.obj")
                remesh_mesh(mesh_path, processed_mesh_path)
            else:
                processed_mesh_path = mesh_path
            
            mesh = trimesh.load(processed_mesh_path)
            if unwrap_mode == "cylindrical": mesh = self.apply_cylindrical_unwrap(mesh)
            elif unwrap_mode == "projective": mesh = self.apply_projective_unwrap(mesh)
            else: mesh = mesh_uv_wrap(mesh)
            mesh.export(unwrapped_obj)

        self.render.load_mesh(mesh=mesh)

        # 3) View Selection
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
            self.config.candidate_view_weights, self.config.max_selected_view_num,
        )

        # 4 & 5) Multiview Diffusion Caching
        alb_raw_dir = os.path.join(dbg_dir, "mv_albedo_raw")
        mr_raw_dir = os.path.join(dbg_dir, "mv_mr_raw")
        
        multiviews_pbr = {"albedo": [], "mr": []}
        
        # If cache exists and we aren't using a manual folder for EVERYTHING, load cache
        if use_cache and os.path.exists(alb_raw_dir) and len(glob.glob(os.path.join(alb_raw_dir, "*.png"))) > 0:
            logger.info("Loading diffusion images from cache...")
            multiviews_pbr["albedo"] = self._load_images_from_dir(alb_raw_dir, "albedo_*.png")
            multiviews_pbr["mr"] = self._load_images_from_dir(mr_raw_dir, "mr_*.png")
        else:
            self.load_models()
            image_prompt = Image.open(image_path).convert("RGB")
            normal_maps = self.view_processor.render_normal_multiview(selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
            position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
            
            res = self.models["multiview_model"](
                [image_prompt.resize((512, 512))],
                normal_maps + position_maps,
                prompt="high quality, realistic skin pores, photorealistic facial details",
                custom_view_size=self.config.resolution,
                resize_input=True,
            )
            multiviews_pbr["albedo"], multiviews_pbr["mr"] = res["albedo"], res["mr"]
            self._save_list_as_images(multiviews_pbr["albedo"], alb_raw_dir, "albedo")
            self._save_list_as_images(multiviews_pbr["mr"], mr_raw_dir, "mr")

        # 6) Preparation for Baking (Manual Override + ESRGAN skipped)
        enhance_images = {"albedo": [], "mr": []}

        # Handle Albedo (Check if we have a manual folder from Qwen/etc)
        if manual_albedo_dir and os.path.exists(manual_albedo_dir):
            logger.info(f"Using manual enhanced albedos from: {manual_albedo_dir}")
            alb_sources = self._load_images_from_dir(manual_albedo_dir)
        else:
            logger.info("No manual albedo dir provided, using raw diffusion.")
            alb_sources = multiviews_pbr["albedo"]

        # Final Resizing Loop (Replacing the ESRGAN logic)
        for i in range(len(alb_sources)):
            # Process Albedo
            alb_img = alb_sources[i].resize((self.config.render_size, self.config.render_size), Image.LANCZOS)
            enhance_images["albedo"].append(alb_img)
            
            # Process MR (Always use raw since ESRGAN is skipped)
            mr_in = multiviews_pbr["mr"][i]
            mr_img = mr_in if isinstance(mr_in, Image.Image) else Image.fromarray(np.asarray(mr_in))
            mr_img = mr_img.resize((self.config.render_size, self.config.render_size), Image.LANCZOS)
            enhance_images["mr"].append(mr_img)

        # 7) Bake
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )

        # 8) Inpaint
        mask_np = (mask.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
        texture_final = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture_final, force_set=True)

        if "mr" in enhance_images:
            texture_mr, _ = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            texture_mr_final = self.view_processor.texture_inpaint(texture_mr, mask_np)
            self.render.set_texture_mr(texture_mr_final)

        # Save outputs
        self.render.save_mesh(output_mesh_path, downsample=True)
        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))

        return output_mesh_path