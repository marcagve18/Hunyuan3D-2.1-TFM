import os
import torch
import trimesh
import numpy as np
import logging
import glob
from PIL import Image

from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.uvwrap_utils import mesh_uv_wrap
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from skin_refine import SkinTextureRefiner

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

        # Skin refinement settings
        self.use_skin_refinement = True
        _here = os.path.dirname(os.path.abspath(__file__))
        self.real_data_dir = "/home/maguilar/TFM/sota_tests/data/real_baby_faces"
        self.stylegan2_D_ckpt = os.path.join(_here, "..", "ckpt", "stylegan2-ffhq-1024x1024.pkl")
        self.refine_steps = 150
        self.refine_lr = 0.002
        self.lr_D = 1e-4  # unused (D is now frozen)


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

    def load_models(self):
        if "multiview_model" in self.models:
            return
        logger.info("Loading AI Models...")
        torch.cuda.empty_cache()
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        if self.config.use_skin_refinement:
            self.models["skin_refiner"] = SkinTextureRefiner(
                D_ckpt_path=self.config.stylegan2_D_ckpt,
                real_data_dir=self.config.real_data_dir,
                num_steps=self.config.refine_steps,
                lr_tex=self.config.refine_lr,
                lr_D=self.config.lr_D,
            )
        logger.info("Models Loaded.")

    def _img_to_tensor(self, img: Image.Image):
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).to(self.config.device)

    def _load_images_from_dir(self, img_dir: str, pattern: str = "*.png"):
        paths = sorted(glob.glob(os.path.join(img_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"No images found in {img_dir} with pattern {pattern}")
        return [Image.open(p).convert("RGB") for p in paths]

    def _apply_baby_skin_material(self, mr_texture: torch.Tensor) -> torch.Tensor:
        """
        Apply baby skin PBR material properties.

        For baby skin:
        - Metallic (R channel) ≈ 0 (non-metallic)
        - Roughness (G channel) relatively high with soft micro-variation
        """
        device = mr_texture.device
        h, w = mr_texture.shape[0], mr_texture.shape[1]

        base_roughness = 0.6
        noise = torch.randn(h, w, device=device) * 0.15
        roughness_variation = torch.clamp(base_roughness + noise, 0.3, 0.85)

        gaussian_kernel_size = 9
        sigma = 3.0
        x = torch.arange(-gaussian_kernel_size // 2 + 1, gaussian_kernel_size // 2 + 1,
                         dtype=torch.float32, device=device)
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1))
        gauss_2d = (gauss_2d / gauss_2d.sum()).unsqueeze(0).unsqueeze(0)

        pad = gaussian_kernel_size // 2
        roughness_padded = torch.nn.functional.pad(
            roughness_variation.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect'
        )
        roughness_smooth = torch.nn.functional.conv2d(roughness_padded, gauss_2d, padding=0)
        roughness_smooth = roughness_smooth.squeeze(0).squeeze(0)

        mr_texture[:, :, 0] = 0.0
        mr_texture[:, :, 1] = roughness_smooth
        return mr_texture

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
        save_glb=False,
        unwrap_mode='cylindrical',
        use_cache=False,
        manual_albedo_dir=None,
        manual_texture_path=None,
        manual_mr_texture_path=None,
    ):
        input_dir = os.path.dirname(mesh_path)
        if output_mesh_path is None:
            output_mesh_path = os.path.join(input_dir, f"textured_{unwrap_mode}.obj")

        inter_dir = os.path.join(os.path.dirname(output_mesh_path), "intermediates")
        dbg_dir = os.path.join(inter_dir, "debug_images")

        # 1 & 2) Mesh & UV
        unwrapped_obj = os.path.join(inter_dir, f"unwrapped_{unwrap_mode}.obj")
        if use_cache and os.path.exists(unwrapped_obj):
            logger.info(f"Loading cached unwrapped mesh: {unwrapped_obj}")
            mesh = trimesh.load(unwrapped_obj)
        else:
            print("not using cache")
            processed_mesh_path = mesh_path
            if use_remesh:
                processed_mesh_path = os.path.join(inter_dir, "step1_remeshed.obj")
                remesh_mesh(mesh_path, processed_mesh_path)

            mesh = trimesh.load(processed_mesh_path)
            if unwrap_mode == "cylindrical":
                mesh = self.apply_cylindrical_unwrap(mesh)
            elif unwrap_mode == "projective":
                mesh = self.apply_projective_unwrap(mesh)
            else:
                mesh = mesh_uv_wrap(mesh)
            mesh.export(unwrapped_obj)

        self.render.load_mesh(mesh=mesh)

        # --- DIRECT TEXTURE APPLICATION BRANCH ---
        if manual_texture_path and os.path.exists(manual_texture_path):
            logger.info(f"Manual Texture detected: {manual_texture_path}. Applying directly.")
            tex_tensor = self._img_to_tensor(Image.open(manual_texture_path).convert("RGB"))
            self.render.set_texture(tex_tensor, force_set=True)
            if manual_mr_texture_path and os.path.exists(manual_mr_texture_path):
                mr_tensor = self._img_to_tensor(Image.open(manual_mr_texture_path).convert("RGB"))
                self.render.set_texture_mr(mr_tensor)
            self.render.save_mesh(output_mesh_path, downsample=True)
            if save_glb:
                convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
            return output_mesh_path
        # ------------------------------------------

        # 3) View Selection
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
            self.config.candidate_view_weights, self.config.max_selected_view_num,
        )

        # 4 & 5) Multiview Diffusion (with caching)
        alb_raw_dir = os.path.join(dbg_dir, "mv_albedo_raw")
        mr_raw_dir = os.path.join(dbg_dir, "mv_mr_raw")
        multiviews_pbr = {"albedo": [], "mr": []}

        if use_cache and os.path.exists(alb_raw_dir) and len(glob.glob(os.path.join(alb_raw_dir, "*.png"))) > 0:
            logger.info("Loading diffusion images from cache...")
            multiviews_pbr["albedo"] = self._load_images_from_dir(alb_raw_dir, "albedo_*.png")
            multiviews_pbr["mr"] = self._load_images_from_dir(mr_raw_dir, "mr_*.png")
        else:
            self.load_models()
            image_prompt = Image.open(image_path).convert("RGB")
            normal_maps = self.view_processor.render_normal_multiview(
                selected_camera_elevs, selected_camera_azims, use_abs_coor=True
            )
            position_maps = self.view_processor.render_position_multiview(
                selected_camera_elevs, selected_camera_azims
            )

            res = self.models["multiview_model"](
                [image_prompt.resize((512, 512))],
                normal_maps + position_maps,
                prompt="high quality, realistic skin pores, photorealistic facial details",
                custom_view_size=self.config.resolution,
                resize_input=True,
            )
            multiviews_pbr["albedo"], multiviews_pbr["mr"] = res["albedo"], res["mr"]

            os.makedirs(alb_raw_dir, exist_ok=True)
            os.makedirs(mr_raw_dir, exist_ok=True)
            for i, im in enumerate(multiviews_pbr["albedo"]):
                im.save(os.path.join(alb_raw_dir, f"albedo_{i:02d}.png"))
            for i, im in enumerate(multiviews_pbr["mr"]):
                im.save(os.path.join(mr_raw_dir, f"mr_{i:02d}.png"))

        # 6) Preparation for Baking
        enhance_images = {"albedo": [], "mr": []}
        alb_sources = self._load_images_from_dir(manual_albedo_dir) if manual_albedo_dir else multiviews_pbr["albedo"]

        for i in range(len(alb_sources)):
            alb_img = alb_sources[i].resize((self.config.render_size, self.config.render_size), Image.LANCZOS)
            enhance_images["albedo"].append(alb_img)
            mr_in = multiviews_pbr["mr"][i]
            mr_img = mr_in if isinstance(mr_in, Image.Image) else Image.fromarray(np.asarray(mr_in))
            enhance_images["mr"].append(
                mr_img.resize((self.config.render_size, self.config.render_size), Image.LANCZOS)
            )

        # 7 & 8) Bake & Inpaint
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
        texture_final = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture_final, force_set=True)

        texture_mr_final = None
        if "mr" in enhance_images:
            texture_mr, _ = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            texture_mr_final = self.view_processor.texture_inpaint(texture_mr, mask_np)
            texture_mr_final = self._apply_baby_skin_material(texture_mr_final)
            self.render.set_texture_mr(texture_mr_final)

        # 9) Skin refinement (optional)
        if self.config.use_skin_refinement:
            self.models["multiview_model"].pipeline.to("cpu")
            torch.cuda.empty_cache()

            image_style = [Image.open(image_path).convert("RGB").resize((512, 512))]
            _refine_debug = os.path.join(os.path.dirname(output_mesh_path), "skin_refiner_debug")
            texture_final, texture_mr_final = self.models["skin_refiner"](
                texture=texture_final,
                texture_mr=texture_mr_final,
                render=self.render,
                reference_images=image_style,
                debug_dir=_refine_debug,
            )
            self.render.set_texture(texture_final, force_set=True)
            self.render.set_texture_mr(texture_mr_final)

        # Save outputs
        self.render.save_mesh(output_mesh_path, downsample=True)
        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))

        return output_mesh_path
