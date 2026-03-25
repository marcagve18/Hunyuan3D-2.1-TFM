"""skin_refine/refiners/codeformer.py — CodeFormer face restoration.
====================================================================
CodeFormer: Towards Robust Face Restoration with Learning Dynamic
Wrapping on Swin Transformer (CVPR 2023)

Advantages over GFPGAN:
  - Better fidelity-restoration balance
  - Swin Transformer backbone for global context
  - More robust to severe degradation
  - Better preserves facial identity
  
Usage:
    refiner = CodeFormerRefiner(ckpt_path="path/to/CodeFormer.pth")
    restored = refiner.restore(face_image)
"""

import os
import logging
from typing import Optional
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

_CODEFORMER_CKPT_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/"
    "CodeFormer.pth"
)


class CodeFormerRefiner(BaseSkinRefiner):
    """CodeFormer-based face restoration.

    Parameters
    ----------
    ckpt_path : str, optional
        Path to CodeFormer.pth. Auto-downloaded if None.
    upscale : int, default=2
        Upscale factor for processing resolution.
    fidelity_weight : float, default=0.5
        Balance between fidelity (0) and quality (1).
        Lower = more restoration, Higher = more faithful to input.
    detection_path : str, optional
        Path to face detection model (yolov5l-face).
    bg_upsampler : str or None, default=None
        Background upsampler model ('realesrgan' or 'gpen').
    device : str, default='cuda'
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        upscale: int = 2,
        fidelity_weight: float = 0.5,
        detection_path: Optional[str] = None,
        bg_upsampler: Optional[str] = None,
        device: str = "cuda",
    ):
        self.upscale = upscale
        self.fidelity_weight = fidelity_weight
        self.detection_path = detection_path
        self.bg_upsampler = bg_upsampler
        self._device = device
        self._restorer = None

        if ckpt_path is None:
            _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_path = os.path.join(_here, "..", "..", "ckpt", "CodeFormer.pth")

        self.ckpt_path = ckpt_path
        self._ensure_checkpoint()

    @property
    def name(self) -> str:
        return "CodeFormer"

    def _ensure_checkpoint(self):
        if not os.path.isfile(self.ckpt_path):
            logger.info(f"[CodeFormer] Downloading checkpoint → {self.ckpt_path}")
            import urllib.request
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            urllib.request.urlretrieve(_CODEFORMER_CKPT_URL, self.ckpt_path)

    def _load_restorer(self):
        if self._restorer is not None:
            return

        logger.info(f"[CodeFormer] Loading from {self.ckpt_path} …")
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
            from basicsr.utils import face_restoration_utils
            from torchvision.transforms.functional import normalize
        except ImportError as e:
            raise RuntimeError(
                "CodeFormer requires basicsr. Install: pip install basicsr"
            ) from e

        self._CodeFormer = CodeFormer
        self._face_utils = face_restoration_utils
        self._normalize = normalize
        self._restorer = True

    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Apply CodeFormer face restoration to a single image.

        Parameters
        ----------
        image : PIL Image (RGB)

        Returns
        -------
        restored : PIL Image (RGB) or None if no face found
        """
        import cv2
        import numpy as np

        self._load_restorer()

        bgr_in = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        bgr_in = bgr_in.astype(np.float32) / 255.0

        restored = self._face_utils.restore_face(
            bgr_in,
            self._CodeFormer(dim=512, n_blocks=12, num_labels=20675),
            upscale=self.upscale,
            only_center_face=False,
            detection_model="retinaface_resnet50",
            bg_upsampler=self.bg_upsampler,
        )

        if restored is None or len(restored) == 0:
            return None

        bgr_out = restored[0]
        bgr_out = np.clip(bgr_out * 255, 0, 255).astype(np.uint8)
        rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)

    def restore_batch(self, images):
        import cv2
        import numpy as np

        self._load_restorer()

        bgr_images = [
            cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
            for img in images
        ]

        results = self._face_utils.restore_face(
            bgr_images,
            self._CodeFormer(dim=512, n_blocks=12, num_labels=20675),
            upscale=self.upscale,
            only_center_face=False,
            detection_model="retinaface_resnet50",
            bg_upsampler=self.bg_upsampler,
        )

        restored = []
        for i, result in enumerate(results):
            if result is None or len(result) == 0:
                restored.append(None)
            else:
                bgr_out = np.clip(result[0] * 255, 0, 255).astype(np.uint8)
                rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
                restored.append(Image.fromarray(rgb_out))
        return restored
