import os
import subprocess
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

import warnings
warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity(50)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Hunyuan3DPaint")

QWEN_CONDA_ENV_PATH = "/gpfs/home/maguilar/miniforge3/envs/qwen_edit"

class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"
        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"

        self.qwen_model_id = "ovedrive/Qwen-Image-Edit-2511-4bit" ##"Qwen/Qwen-Image-Edit-2511"
        self.qwen_dtype = "bf16"
        self.qwen_num_inference_steps = 24
        self.qwen_true_cfg_scale = 4.0
        self.qwen_guidance_scale = 1.0
        self.qwen_seed = 1234
        self.qwen_cli_path = "/home/maguilar/TFM/Hunyuan3D-2.1/hy3dpaint/utils/qwen_edit_worker.py"
        self.qwen_env_path = QWEN_CONDA_ENV_PATH

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
        # Note: load_models() is now called lazily only if cache is missing

    def load_models(self):
        if "multiview_model" in self.models:
            return
        logger.info("Loading AI Models (Inference required)...")
        torch.cuda.empty_cache()
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

    def _save_pil(self, img: Image.Image, path: str):
        self._ensure_dir(os.path.dirname(path))
        img.save(path)

    def _save_np_image(self, arr: np.ndarray, path: str):
        self._ensure_dir(os.path.dirname(path))
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if arr.ndim == 4: arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))
        
        if arr.ndim == 2: im = Image.fromarray(self._to_uint8(arr), mode="L")
        else: im = Image.fromarray(self._to_uint8(arr[:, :, :3]), mode="RGB")
        im.save(path)

    def _save_list_as_images(self, imgs, out_dir: str, prefix: str):
        self._ensure_dir(out_dir)
        for i, im in enumerate(imgs):
            out_path = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
            if isinstance(im, Image.Image): self._save_pil(im, out_path)
            else: self._save_np_image(im, out_path)

    def _load_images_from_dir(self, img_dir: str, prefix: str):
        paths = sorted(glob.glob(os.path.join(img_dir, f"{prefix}_*.png")))
        return [Image.open(p).convert("RGB") for p in paths]

    def _run_qwen_albedo_batch(self, *, albedo_in_dir: str, albedo_out_dir: str, image_prompt_path: str, use_cache=True):
        # Cache check for Qwen outputs
        if use_cache and os.path.exists(albedo_out_dir) and len(glob.glob(os.path.join(albedo_out_dir, "*.png"))) > 0:
            logger.info(f"Found cached Qwen results in {albedo_out_dir}. Skipping subprocess.")
            return

        self._ensure_dir(albedo_out_dir)
        env_path = self.config.qwen_env_path
        cli_path = self.config.qwen_cli_path

        # Log GPU memory before running
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU Memory before Qwen: {gpu_mem_allocated:.2f}GB allocated, "
                        f"{gpu_mem_reserved:.2f}GB reserved, {gpu_mem_total:.2f}GB total")
            else:
                logger.info("GPU Memory: CUDA not available")
        except Exception as e:
            logger.warning(f"Could not log GPU memory: {e}")

        cmd = [
            "conda", "run", "-p", env_path, "python", cli_path,
            "--model_id", self.config.qwen_model_id,
            "--input_dir", albedo_in_dir,
            "--output_dir", albedo_out_dir,
            "--prompt_image", image_prompt_path,
            "--glob", "albedo_*.png",
            "--dtype", self.config.qwen_dtype,
            "--num_inference_steps", str(self.config.qwen_num_inference_steps),
            "--true_cfg_scale", str(self.config.qwen_true_cfg_scale),
            "--guidance_scale", str(self.config.qwen_guidance_scale),
            "--seed", str(self.config.qwen_seed),
        ]

        logger.info("Running Qwen enhancement subprocess...")
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1  # Line buffered
        )

        # Stream output line by line
        for line in proc.stdout:
            logger.info(f"[Qwen] {line.rstrip()}")

        proc.wait()

        if proc.returncode != 0:
            logger.error(f"Qwen subprocess failed with code {proc.returncode}")

    @torch.no_grad()
    def __call__(
        self,
        mesh_path=None,
        image_path=None,
        output_mesh_path=None,
        use_remesh=False,
        save_glb=True,
        unwrap_mode='cylindrical',
        use_cache=True # Added cache toggle
    ):
        input_dir = os.path.dirname(mesh_path)
        if output_mesh_path is None:
            output_mesh_path = os.path.join(input_dir, f"textured_{unwrap_mode}.obj")

        inter_dir = os.path.join(os.path.dirname(output_mesh_path), "intermediates")
        dbg_dir = os.path.join(inter_dir, "debug_images")
        
        # Define specific cache paths
        albedo_raw_dir = os.path.join(dbg_dir, "mv_albedo_raw")
        mr_raw_dir = os.path.join(dbg_dir, "mv_mr_raw")
        albedo_qwen_dir = os.path.join(dbg_dir, "mv_albedo_qwen")
        input_prompt_path = os.path.join(dbg_dir, "input_image.png")

        # 1 & 2) Mesh/UV (Check if already unwrapped)
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

        # 3) View Selection (Required for Bake logic, very fast)
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
            self.config.candidate_view_weights, self.config.max_selected_view_num,
        )

        # 4 & 5) Multi-view Diffusion Logic
        multiviews_pbr = {"albedo": [], "mr": []}
        cache_exists = os.path.exists(albedo_raw_dir) and len(glob.glob(os.path.join(albedo_raw_dir, "*.png"))) > 0

        if use_cache and cache_exists:
            logger.info("Found cached diffusion images. Skipping diffusion model.")
            multiviews_pbr["albedo"] = self._load_images_from_dir(albedo_raw_dir, "albedo")
            multiviews_pbr["mr"] = self._load_images_from_dir(mr_raw_dir, "mr")
        else:
            self.load_models() # Only load if we actually need to run inference
            image_prompt = Image.open(image_path).convert("RGB")
            self._save_pil(image_prompt, input_prompt_path)
            
            normal_maps = self.view_processor.render_normal_multiview(selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
            position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
            
            res = self.models["multiview_model"](
                [image_prompt.resize((512, 512))],
                normal_maps + position_maps,
                prompt="high quality, realistic skin pores, photorealistic facial details",
                custom_view_size=self.config.resolution,
                resize_input=True,
            )
            multiviews_pbr["albedo"] = res["albedo"]
            multiviews_pbr["mr"] = res["mr"]
            
            self._save_list_as_images(multiviews_pbr["albedo"], albedo_raw_dir, "albedo")
            self._save_list_as_images(multiviews_pbr["mr"], mr_raw_dir, "mr")

        # 6) Qwen Enhancement
        self._run_qwen_albedo_batch(
            albedo_in_dir=albedo_raw_dir,
            albedo_out_dir=albedo_qwen_dir,
            image_prompt_path=input_prompt_path,
            use_cache=use_cache
        )

        # Re-assemble enhanced images
        enhance_images = {"albedo": [], "mr": []}
        for i in range(len(multiviews_pbr["albedo"])):
            alb_q = Image.open(os.path.join(albedo_qwen_dir, f"albedo_{i:02d}.png")).convert("RGB")
            enhance_images["albedo"].append(alb_q.resize((self.config.render_size, self.config.render_size)))
            
            mr_in = multiviews_pbr["mr"][i]
            mr_img = mr_in if isinstance(mr_in, Image.Image) else Image.fromarray(np.asarray(mr_in))
            enhance_images["mr"].append(mr_img.resize((self.config.render_size, self.config.render_size)))

        # 7) Final Bake & Inpaint (Standard logic)
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
        texture_final = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture_final, force_set=True)

        if "mr" in enhance_images:
            texture_mr, _ = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            texture_mr_final = self.view_processor.texture_inpaint(texture_mr, mask_np)
            self.render.set_texture_mr(texture_mr_final)

        self.render.save_mesh(output_mesh_path, downsample=True)
        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))

        logger.info(f"Pipeline finished. Output: {output_mesh_path}")
        return output_mesh_path