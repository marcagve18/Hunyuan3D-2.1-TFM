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
from utils.uvwrap_utils import mesh_uv_wrap
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from skin_refine import SkinTextureRefiner

import warnings
warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity(50)
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL, EulerDiscreteScheduler

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
        self.sd15_pretrained_path = "marcagve18/baby-face-generation"
        self.sd15_vae_path = "stabilityai/sd-vae-ft-ema"
        self.sd15_controlnet_path = "lllyasviel/control_v11f1e_sd15_tile"
        self.sd15_refine_strength = 0.12
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

        # Skin texture refinement (GFPGAN-based)
        self.use_skin_refinement        = True
        _here = os.path.dirname(os.path.abspath(__file__))
        self.skin_refine_ckpt           = os.path.join(_here, "..", "ckpt", "GFPGANv1.4.pth")
        self.skin_refine_strength       = 0.6   # GFPGAN weight: 0=original, 1=fully restored
        self.skin_refine_num_passes     = 2     # render→GFPGAN→bake iterations
        self.skin_refine_blend_alpha    = 0.8   # UV blend with original texture
        self.skin_refine_grain          = 0.018 # luminance grain std-dev (0=off)
        self.skin_refine_resolution     = 512

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
            logger.info("Loading SkinTextureRefiner (GFPGAN) …")
            self.models["skin_refiner"] = SkinTextureRefiner(
                ckpt_path             = self.config.skin_refine_ckpt,
                upscale               = 2,
                restoration_strength  = self.config.skin_refine_strength,
                num_passes            = self.config.skin_refine_num_passes,
                blend_alpha           = self.config.skin_refine_blend_alpha,
                grain_strength        = self.config.skin_refine_grain,
                refine_resolution     = self.config.skin_refine_resolution,
                device                = self.config.device,
            )
        
        logger.info("Models Loaded.")

    def refine_with_sd15(self, images, prompt="high quality, photorealistic, detailed texture"):
        if "sd15_model" not in self.models:
            logger.info("Loading ControlNet model...")
            self.models["controlnet"] = ControlNetModel.from_pretrained(
                self.config.sd15_controlnet_path,
                torch_dtype=torch.float16,
            )
            
            logger.info("Loading VAE...")
            self.models["vae"] = AutoencoderKL.from_pretrained(
                self.config.sd15_vae_path,
                torch_dtype=torch.float16,
            )
            
            logger.info("Loading SD1.5 refinement model with ControlNet...")
            self.models["sd15_model"] = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.config.sd15_pretrained_path,
                vae=self.models["vae"],
                torch_dtype=torch.float16,
                controlnet=self.models["controlnet"],
            )
            self.models["sd15_model"].to(self.config.device)
            self.models["sd15_model"].safety_checker = None
            self.models["sd15_model"].enable_xformers_memory_efficient_attention()
            self.models["sd15_model"].scheduler = EulerDiscreteScheduler.from_config(
                self.models["sd15_model"].scheduler.config
            )
        
        negative_prompt = "teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant"
        
        refined_images = []
        for i, img in enumerate(images):
            logger.info(f"Refining view {i+1}/{len(images)} with SD1.5 ControlNet...")
            
            original_size = img.size
            img_upscaled = img.resize((1024, 1024), Image.LANCZOS)
            
            generator = torch.manual_seed(42)
            
            refined = self.models["sd15_model"](
                prompt="Portrait of a white baby smiling",
                image=img_upscaled,
                control_image=img_upscaled,
                strength=self.config.sd15_refine_strength,
                controlnet_conditioning_scale=0.9,
                negative_prompt=negative_prompt,
                guidance_scale=5,
                generator=generator,
                num_inference_steps=50,
                guess_mode=False,
            ).images[0]
            
            refined_downscaled = refined.resize(original_size, Image.LANCZOS)
            refined_images.append(refined_downscaled)
        return refined_images

    def _img_to_tensor(self, img: Image.Image):
        """Helper to convert PIL Image to the float32 tensor the renderer expects."""
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).to(self.config.device)

    def _ensure_dir(self, p: str):
        os.makedirs(p, exist_ok=True)

    def _load_images_from_dir(self, img_dir: str, pattern: str = "*.png"):
        paths = sorted(glob.glob(os.path.join(img_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"No images found in {img_dir} with pattern {pattern}")
        return [Image.open(p).convert("RGB") for p in paths]

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
        manual_mr_texture_path=None  
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
            if unwrap_mode == "cylindrical": mesh = self.apply_cylindrical_unwrap(mesh)
            elif unwrap_mode == "projective": mesh = self.apply_projective_unwrap(mesh)
            else: mesh = mesh_uv_wrap(mesh)
            mesh.export(unwrapped_obj)

        self.render.load_mesh(mesh=mesh)

        # --- DIRECT TEXTURE APPLICATION BRANCH ---
        if manual_texture_path and os.path.exists(manual_texture_path):
            logger.info(f"Manual Texture detected: {manual_texture_path}. Applying directly.")
            
            # Convert PIL to Tensor before calling set_texture
            tex_pil = Image.open(manual_texture_path).convert("RGB")
            tex_tensor = self._img_to_tensor(tex_pil)
            self.render.set_texture(tex_tensor, force_set=True)
            
            if manual_mr_texture_path and os.path.exists(manual_mr_texture_path):
                mr_pil = Image.open(manual_mr_texture_path).convert("RGB")
                mr_tensor = self._img_to_tensor(mr_pil)
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

        # 4 & 5) Multiview Diffusion Caching
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
            
            # SD1.5 refinement disabled
            alb_refined_dir = os.path.join(dbg_dir, "mv_albedo_refined")
            
            os.makedirs(alb_raw_dir, exist_ok=True)
            os.makedirs(mr_raw_dir, exist_ok=True)
            for i, im in enumerate(multiviews_pbr["albedo"]): im.save(os.path.join(alb_raw_dir, f"albedo_{i:02d}.png"))
            for i, im in enumerate(multiviews_pbr["mr"]): im.save(os.path.join(mr_raw_dir, f"mr_{i:02d}.png"))

        # 6) Preparation for Baking
        enhance_images = {"albedo": [], "mr": []}
        alb_sources = self._load_images_from_dir(manual_albedo_dir) if manual_albedo_dir else multiviews_pbr["albedo"]

        for i in range(len(alb_sources)):
            alb_img = alb_sources[i].resize((self.config.render_size, self.config.render_size), Image.LANCZOS)
            enhance_images["albedo"].append(alb_img)
            mr_in = multiviews_pbr["mr"][i]
            mr_img = mr_in if isinstance(mr_in, Image.Image) else Image.fromarray(np.asarray(mr_in))
            enhance_images["mr"].append(mr_img.resize((self.config.render_size, self.config.render_size), Image.LANCZOS))

        # 7 & 8) Bake & Inpaint
        texture, mask = self.view_processor.bake_from_multiview(enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights)
        mask_np = (mask.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
        texture_final = self.view_processor.texture_inpaint(texture, mask_np)
        
        # texture_final is usually a tensor here from the inpaint process
        self.render.set_texture(texture_final, force_set=True)

        if "mr" in enhance_images:
            texture_mr, _ = self.view_processor.bake_from_multiview(enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights)
            texture_mr_final = self.view_processor.texture_inpaint(texture_mr, mask_np)
            self.render.set_texture_mr(texture_mr_final)

        # 9) Skin texture refinement (GFPGAN face restoration)
        if self.config.use_skin_refinement and "skin_refiner" in self.models:
            # Free multiview model GPU memory before running GFPGAN
            self.models["multiview_model"].pipeline.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Running GFPGAN skin texture refinement …")

            _debug = os.path.join(os.path.dirname(output_mesh_path), "skin_refiner_debug")
            self.models["skin_refiner"](
                render    = self.render,
                debug_dir = _debug,
            )

        # Debug: save front-view renders before and after refinement for quick inspection
        _snap_dir = os.path.join(os.path.dirname(output_mesh_path), "snapshots")
        os.makedirs(_snap_dir, exist_ok=True)
        for _elev, _azim, _tag in [(0, 0, "front"), (0, 90, "right"), (0, -90, "left"), (20, 0, "top")]:
            _render_snap = self.render.render_normal(_elev, _azim, return_type="pl")
            _render_snap.save(os.path.join(_snap_dir, f"normal_{_tag}.png"))
        logger.info(f"Snapshots saved to {_snap_dir}")

        # Save outputs
        self.render.save_mesh(output_mesh_path, downsample=True)
        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))

        return output_mesh_path