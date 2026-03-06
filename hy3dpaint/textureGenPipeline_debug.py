import os
import torch
import trimesh
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image

from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.image_super_utils import imageSuperNet
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
        self.load_models()

    def load_models(self):
        logger.info("Loading AI Models...")
        torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        logger.info("Models Loaded.")

    # ---------------------------
    # Image-only debug helpers
    # ---------------------------
    def _ensure_dir(self, p: str):
        os.makedirs(p, exist_ok=True)

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        a = arr
        if np.issubdtype(a.dtype, np.floating):
            a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
            mn, mx = float(a.min()), float(a.max())
            # If likely [0,1], scale; else normalize for visibility
            if 0.0 <= mn and mx <= 1.0:
                a = a * 255.0
            else:
                if mx - mn > 1e-8:
                    a = (a - mn) / (mx - mn) * 255.0
                else:
                    a = np.zeros_like(a)
        return np.clip(a, 0, 255).astype(np.uint8)

    def _save_np_image(self, arr: np.ndarray, path: str):
        self._ensure_dir(os.path.dirname(path))
        if isinstance(arr, torch.Tensor):
            arr = arr.detach()
            if arr.is_cuda:
                arr = arr.cpu()
            arr = arr.numpy()

        # Handle BCHW / CHW / HWC / HW
        if arr.ndim == 4:  # BCHW
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW -> HWC
            arr = np.transpose(arr[:3], (1, 2, 0))

        if arr.ndim == 2:
            im = Image.fromarray(self._to_uint8(arr), mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            im = Image.fromarray(self._to_uint8(arr[:, :, 0]), mode="L")
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            im = Image.fromarray(self._to_uint8(arr[:, :, :3]), mode="RGB")
        else:
            raise ValueError(f"Unsupported array shape for image save: {arr.shape}")
        im.save(path)

    def _save_pil(self, img: Image.Image, path: str):
        self._ensure_dir(os.path.dirname(path))
        img.save(path)

    def _save_list_as_images(self, imgs, out_dir: str, prefix: str):
        self._ensure_dir(out_dir)
        for i, im in enumerate(imgs):
            # Many pieces are PIL already; fall back to numpy/tensor saver
            out_path = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
            if isinstance(im, Image.Image):
                self._save_pil(im, out_path)
            else:
                self._save_np_image(im, out_path)

    # ---------------------------
    # UV visualization
    # ---------------------------
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
    ):
        input_dir = os.path.dirname(mesh_path)
        if output_mesh_path is None:
            output_mesh_path = os.path.join(input_dir, f"textured_{unwrap_mode}.obj")

        inter_dir = os.path.join(os.path.dirname(output_mesh_path), "intermediates")
        dbg_dir = os.path.join(inter_dir, "debug_images")
        os.makedirs(inter_dir, exist_ok=True)
        os.makedirs(dbg_dir, exist_ok=True)

        # 1) Mesh preprocess
        if use_remesh:
            processed_mesh_path = os.path.join(inter_dir, "step1_remeshed.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        # 2) UV unwrap
        mesh = trimesh.load(processed_mesh_path)
        if unwrap_mode == "cylindrical":
            mesh = self.apply_cylindrical_unwrap(mesh)
        elif unwrap_mode == "projective":
            mesh = self.apply_projective_unwrap(mesh)
        else:
            mesh = mesh_uv_wrap(mesh)

        self._save_uv_visualization(mesh, os.path.join(dbg_dir, f"uv_layout_{unwrap_mode}.png"))
        mesh.export(os.path.join(inter_dir, f"unwrapped_{unwrap_mode}.obj"))
        self.render.load_mesh(mesh=mesh)

        # 3) View selection + render normals/positions
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(
            selected_camera_elevs, selected_camera_azims
        )

        self._save_list_as_images(normal_maps, os.path.join(dbg_dir, "normal_maps"), "normal")
        self._save_list_as_images(position_maps, os.path.join(dbg_dir, "position_maps"), "position")

        # 4) Load and save input images (original + resized)
        image_prompt = Image.open(image_path).convert("RGB")
        self._save_pil(image_prompt, os.path.join(dbg_dir, "input_image.png"))

        image_style = [image_prompt.resize((512, 512))]
        self._save_pil(image_style[0], os.path.join(dbg_dir, "input_image_512.png"))

        # 5) Multiview diffusion (save outputs)
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt="high quality, realistic skin pores, photorealistic facial details",
            custom_view_size=self.config.resolution,
            resize_input=True,
        )

        # Save raw diffusion outputs
        if "albedo" in multiviews_pbr:
            self._save_list_as_images(multiviews_pbr["albedo"], os.path.join(dbg_dir, "mv_albedo_raw"), "albedo")
        if "mr" in multiviews_pbr:
            self._save_list_as_images(multiviews_pbr["mr"], os.path.join(dbg_dir, "mv_mr_raw"), "mr")

        # 6) Super-res + resize (save)
        enhance_images = {"albedo": [], "mr": []}
        for i in range(len(multiviews_pbr["albedo"])):
            alb_sr = self.models["super_model"](multiviews_pbr["albedo"][i])
            self._save_pil(alb_sr, os.path.join(dbg_dir, "mv_albedo_super", f"albedo_super_{i:02d}.png"))

            alb_rs = alb_sr.resize((self.config.render_size, self.config.render_size))
            self._save_pil(alb_rs, os.path.join(dbg_dir, "mv_albedo_super_resized", f"albedo_super_resized_{i:02d}.png"))
            enhance_images["albedo"].append(alb_rs)

            mr_sr = self.models["super_model"](multiviews_pbr["mr"][i])
            self._save_pil(mr_sr, os.path.join(dbg_dir, "mv_mr_super", f"mr_super_{i:02d}.png"))

            mr_rs = mr_sr.resize((self.config.render_size, self.config.render_size))
            self._save_pil(mr_rs, os.path.join(dbg_dir, "mv_mr_super_resized", f"mr_super_resized_{i:02d}.png"))
            enhance_images["mr"].append(mr_rs)

        # 7) Bake (save baked texture + mask)
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )

        # Try saving baked texture in the most permissive way:
        # - If PIL: save directly
        # - Else: convert via numpy/tensor helper
        if isinstance(texture, Image.Image):
            self._save_pil(texture, os.path.join(dbg_dir, "baked_texture_raw.png"))
        else:
            self._save_np_image(texture, os.path.join(dbg_dir, "baked_texture_raw.png"))

        # Mask usually tensor HxWx1 or HxW (0..1)
        self._save_np_image(mask, os.path.join(dbg_dir, "baked_mask_raw.png"))

        # 8) Inpaint final (save final textures)
        mask_np = (mask.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
        texture_final = self.view_processor.texture_inpaint(texture, mask_np)

        if isinstance(texture_final, Image.Image):
            self._save_pil(texture_final, os.path.join(dbg_dir, "texture_final_inpaint.png"))
        else:
            self._save_np_image(texture_final, os.path.join(dbg_dir, "texture_final_inpaint.png"))

        self.render.set_texture(texture_final, force_set=True)

        # MR branch (save)
        if "mr" in enhance_images:
            texture_mr, _ = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )

            if isinstance(texture_mr, Image.Image):
                self._save_pil(texture_mr, os.path.join(dbg_dir, "baked_mr_raw.png"))
            else:
                self._save_np_image(texture_mr, os.path.join(dbg_dir, "baked_mr_raw.png"))

            texture_mr_final = self.view_processor.texture_inpaint(texture_mr, mask_np)
            if isinstance(texture_mr_final, Image.Image):
                self._save_pil(texture_mr_final, os.path.join(dbg_dir, "mr_final_inpaint.png"))
            else:
                self._save_np_image(texture_mr_final, os.path.join(dbg_dir, "mr_final_inpaint.png"))

            self.render.set_texture_mr(texture_mr_final)

        # Save mesh
        self.render.save_mesh(output_mesh_path, downsample=True)
        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))

        return output_mesh_path
