"""skin_refine/refiners/sd.py — Stable Diffusion face/skin refinement.
=====================================================================
Uses SD 1.5 with ControlNet img2img for photorealistic face enhancement.

Advantages:
  - Diffusion-based → more realistic skin tones and pores
  - Can be guided with prompts for style control
  - Better at generating fine texture details
  
Uses the same negative prompts and settings as refine_with_sd15.
"""

import os
import logging
import torch
from typing import Optional, List
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_SD15_PRETRAINED = "marcagve18/baby-face-generation"
_SD15_VAE = "stabilityai/sd-vae-ft-ema"
_SD15_CONTROLNET = "lllyasviel/control_v11f1e_sd15_tile"


class SDRefinerRefiner(BaseSkinRefiner):
    """SD 1.5 ControlNet img2img face/skin refiner.

    Parameters
    ----------
    pretrained_path : str, default='marcagve18/baby-face-generation'
        SD 1.5 model path on HuggingFace.
    vae_path : str, default='stabilityai/sd-vae-ft-ema'
        VAE model path.
    controlnet_path : str, default='lllyasviel/control_v11f1e_sd15_tile'
        ControlNet model path.
    prompt : str, default='Portrait of a white baby smiling'
        Positive prompt for generation.
    negative_prompt : str, optional
        Negative prompt. Uses built-in anti-artifact prompts if None.
    strength : float, default=0.12
        img2img strength (0=keep original, 1=full regeneration).
    guidance_scale : float, default=5.0
        CFG scale.
    num_inference_steps : int, default=50
        Sampling steps.
    seed : int, default=42
        RNG seed for reproducibility.
    upscale_to : int, default=1024
        Resolution to upscale to before SD processing.
    device : str, default='cuda'
    """

    def __init__(
        self,
        pretrained_path: str = _SD15_PRETRAINED,
        vae_path: str = _SD15_VAE,
        controlnet_path: str = _SD15_CONTROLNET,
        prompt: str = "Portrait of a white baby smiling",
        negative_prompt: Optional[str] = None,
        strength: float = 0.12,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: int = 42,
        upscale_to: int = 1024,
        device: str = "cuda",
    ):
        self.pretrained_path = pretrained_path
        self.vae_path = vae_path
        self.controlnet_path = controlnet_path
        self.prompt = prompt
        self.negative_prompt = negative_prompt or (
            "teeth, tooth, open mouth, longbody, lowres, bad anatomy, "
            "bad hands, missing fingers, extra digit, fewer digits, "
            "cropped, worst quality, low quality, mutant"
        )
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.upscale_to = upscale_to
        self._device = device
        self._pipeline = None
        self._controlnet = None
        self._vae = None

    @property
    def name(self) -> str:
        return "SD-1.5-ControlNet"

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        logger.info("[SDRefiner] Loading ControlNet…")
        from diffusers import ControlNetModel, AutoencoderKL
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        from diffusers.schedulers import EulerDiscreteScheduler

        self._controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path, torch_dtype=torch.float16
        )

        logger.info("[SDRefiner] Loading VAE…")
        self._vae = AutoencoderKL.from_pretrained(
            self.vae_path, torch_dtype=torch.float16
        )

        logger.info("[SDRefiner] Loading SD 1.5 refinement model…")
        self._pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.pretrained_path,
            vae=self._vae,
            torch_dtype=torch.float16,
            controlnet=self._controlnet,
        )
        self._pipeline.to(self._device)
        self._pipeline.safety_checker = None
        self._pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self._pipeline.scheduler.config
        )
        logger.info("[SDRefiner] Ready.")

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Apply SD-based face/skin refinement.

        Parameters
        ----------
        image : PIL Image (RGB)

        Returns
        -------
        refined : PIL Image (RGB) or None on failure
        """
        self._load_pipeline()

        try:
            original_size = image.size
            img_upscaled = image.resize(
                (self.upscale_to, self.upscale_to), Image.LANCZOS
            )

            generator = torch.Generator(device=self._device).manual_seed(self.seed)

            refined = self._pipeline(
                prompt=self.prompt,
                image=img_upscaled,
                control_image=img_upscaled,
                strength=self.strength,
                controlnet_conditioning_scale=0.9,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guess_mode=False,
            ).images[0]

            refined_downscaled = refined.resize(original_size, Image.LANCZOS)
            return refined_downscaled

        except Exception as e:
            logger.warning(f"[SDRefiner] Refinement failed: {e}")
            return None

    def restore_batch(self, images: List[Image.Image]) -> List[Optional[Image.Image]]:
        """Apply SD refinement to a batch of images."""
        self._load_pipeline()
        results = []
        for i, img in enumerate(images):
            logger.info(f"[SDRefiner] Processing {i+1}/{len(images)}…")
            results.append(self.restore(img))
        return results

    def to(self, device: str):
        """Move pipeline to device."""
        self._device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
        return self
