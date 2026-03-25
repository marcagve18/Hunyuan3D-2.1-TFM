"""skin_refine/refiners/gfpgan.py — GFPGAN face restoration.
=============================================================
GFPGAN: Towards Real-World Blind Face Restoration with Generative
Facial Prior (CVPR 2021)

Kept for compatibility; CodeFormer is recommended for better quality.
"""

import os
import logging
from typing import Optional
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_GFPGAN_CKPT_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.4/GFPGANv1.4.pth"
)


class GFPGANRefiner(BaseSkinRefiner):
    """GFPGAN-based face restoration.

    Parameters
    ----------
    ckpt_path : str, optional
        Path to GFPGANv1.4.pth. Auto-downloaded if None.
    upscale : int, default=2
        Internal upscale factor for processing.
    arch : str, default='clean'
        GFPGAN architecture ('clean' or 'gfpgan' variant).
    channel_multiplier : int, default=2
        Channel width multiplier.
    restoration_strength : float, default=0.6
        Restoration weight [0,1]. 0=original, 1=fully restored.
    bg_upsampler : object, optional
        Background upsampler (e.g., Real-ESRGAN).
    device : str, default='cuda'
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        upscale: int = 2,
        arch: str = "clean",
        channel_multiplier: int = 2,
        restoration_strength: float = 0.6,
        bg_upsampler=None,
        device: str = "cuda",
    ):
        self.upscale = upscale
        self.arch = arch
        self.channel_multiplier = channel_multiplier
        self.restoration_strength = restoration_strength
        self.bg_upsampler = bg_upsampler
        self._device = device
        self._restorer = None

        if ckpt_path is None:
            _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_path = os.path.join(_here, "..", "..", "ckpt", "GFPGANv1.4.pth")

        self.ckpt_path = ckpt_path
        self._ensure_checkpoint()

    @property
    def name(self) -> str:
        return "GFPGAN"

    def _ensure_checkpoint(self):
        if not os.path.isfile(self.ckpt_path):
            logger.info(f"[GFPGAN] Downloading checkpoint → {self.ckpt_path}")
            import urllib.request
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            urllib.request.urlretrieve(_GFPGAN_CKPT_URL, self.ckpt_path)

    def _load_restorer(self):
        if self._restorer is not None:
            return

        logger.info(f"[GFPGAN] Loading from {self.ckpt_path} …")
        from gfpgan import GFPGANer

        self._restorer = GFPGANer(
            model_path=self.ckpt_path,
            upscale=self.upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=self.bg_upsampler,
        )
        logger.info("[GFPGAN] Ready.")

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Apply GFPGAN face restoration to a single image.

        Parameters
        ----------
        image : PIL Image (RGB, uint8)

        Returns
        -------
        restored : PIL Image (RGB) or None if no face found
        """
        import cv2
        import numpy as np

        self._load_restorer()

        np_rgb = np.array(image.convert("RGB"))
        bgr_in = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        _, _, bgr_out = self._restorer.enhance(
            bgr_in,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=self.restoration_strength,
        )

        if bgr_out is None:
            return None

        rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)
