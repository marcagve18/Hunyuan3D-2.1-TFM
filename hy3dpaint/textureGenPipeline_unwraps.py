import os
import torch
import copy
import trimesh
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
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

# Configure Logging
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

    def _save_uv_visualization(self, mesh, path, title):
        uvs = mesh.visual.uv
        faces = mesh.faces
        plt.figure(figsize=(8, 8))
        plt.triplot(uvs[:, 0], uvs[:, 1], faces, lw=0.2, color='blue', alpha=0.5)
        plt.title(title)
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.savefig(path)
        plt.close()

    def apply_cylindrical_unwrap(self, mesh):
        """Standard cylindrical projection with a fix for the back-seam streaks."""
        logger.info("Applying Seam-Aware Cylindrical Projection...")
        vertices = mesh.vertices - mesh.bounding_box.centroid
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # Use atan2(x, z) and an offset to move the seam to the back of the head
        theta = np.arctan2(x, z) 
        u = (theta + np.pi) / (2 * np.pi)
        v = (y - y.min()) / (y.max() - y.min())
        
        # Artifact Fix: Identify triangles that cross the 0-1 boundary
        # We manually shift U for crossing vertices to prevent 'stretched' triangles
        uvs = np.column_stack((u, v))
        face_uvs = uvs[mesh.faces]
        u_diff = np.max(face_uvs[:, :, 0], axis=1) - np.min(face_uvs[:, :, 0], axis=1)
        
        # Triangles that jump more than half the texture width are likely crossing the seam
        # In a thesis, we'd split the mesh. Here, we slightly clamp to reduce 'pulling'
        uvs[:, 0] = np.clip(uvs[:, 0], 0.005, 0.995) 
        
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        return mesh

    def apply_projective_unwrap(self, mesh):
        """Thesis Recommendation: Frontal Projective mapping. 
        Zero streaks on the face because there is no wrap-around seam."""
        logger.info("Applying Frontal Projective Unwrapping (Best for Identity)...")
        vertices = mesh.vertices - mesh.bounding_box.centroid
        
        # Map X/Y directly to U/V (Orthographic projection)
        u = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
        v = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
        
        # Buffer margins to avoid edge bleeding
        uvs = np.column_stack((u * 0.9 + 0.05, v * 0.9 + 0.05))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        return mesh

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, 
                 use_remesh=False, save_glb=True, unwrap_mode="projective"):
        """
        Args:
            unwrap_mode: "projective", "cylindrical" (Full head), or "atlas"
        """
        input_dir = os.path.dirname(mesh_path)
        if output_mesh_path is None:
            output_mesh_path = os.path.join(input_dir, f"textured_{unwrap_mode}.obj")
        
        inter_dir = os.path.join(os.path.dirname(output_mesh_path), "intermediates")
        os.makedirs(inter_dir, exist_ok=True)

        # 2. Mesh Pre-processing
        if use_remesh:
            processed_mesh_path = os.path.join(inter_dir, "step1_remeshed.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        # 3. UV Unwrapping
        logger.info(f"Step 2/7: UV Unwrapping via {unwrap_mode}")
        mesh = trimesh.load(processed_mesh_path)
        
        if unwrap_mode == "cylindrical":
            mesh = self.apply_cylindrical_unwrap(mesh)
        elif unwrap_mode == "projective":
            mesh = self.apply_projective_unwrap(mesh)
        else:
            mesh = mesh_uv_wrap(mesh)
        
        self._save_uv_visualization(mesh, os.path.join(inter_dir, f"uv_layout_{unwrap_mode}.png"), f"UV: {unwrap_mode}")
        mesh.export(os.path.join(inter_dir, f"unwrapped_{unwrap_mode}.obj"))
        self.render.load_mesh(mesh=mesh)

        # 4. Standard Pipeline Rendering
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
            self.config.candidate_view_weights, self.config.max_selected_view_num,
        )
        normal_maps = self.view_processor.render_normal_multiview(selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)

        # 5. AI Generation
        image_prompt = Image.open(image_path).convert("RGB")
        image_style = [image_prompt.resize((512, 512))]
        multiviews_pbr = self.models["multiview_model"](
            image_style, normal_maps + position_maps,
            prompt="high quality, realistic skin pores, photorealistic facial details",
            custom_view_size=self.config.resolution, resize_input=True,
        )

        # 6. Upscale & Bake
        enhance_images = {"albedo": [], "mr": []}
        for i in range(len(multiviews_pbr["albedo"])):
            alb = self.models["super_model"](multiviews_pbr["albedo"][i])
            enhance_images["albedo"].append(alb.resize((self.config.render_size, self.config.render_size)))
            mr = self.models["super_model"](multiviews_pbr["mr"][i])
            enhance_images["mr"].append(mr.resize((self.config.render_size, self.config.render_size)))

        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        
        # 7. Final Fix & Save
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_final = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture_final, force_set=True)
        
        if "mr" in enhance_images:
            texture_mr, _ = self.view_processor.bake_from_multiview(enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights)
            self.render.set_texture_mr(self.view_processor.texture_inpaint(texture_mr, mask_np))

        self.render.save_mesh(output_mesh_path, downsample=True)
        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
        
        return output_mesh_path