"""skin_refine/base.py — Abstract base class for skin refiners.
===================================================================
Strategy pattern: each refiner implements face/skin enhancement and
can be swapped at runtime via the SkinTextureRefiner orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import torch
import numpy as np
from PIL import Image


class BaseSkinRefiner(ABC):
    """Abstract interface for face/skin restoration strategies.
    
    All refiners must implement:
      1. restore(image) → PIL Image or np.ndarray
      2. Optional: restore_batch(images) → list of restored images
      
    The orchestrator (SkinTextureRefiner) handles multi-view rendering,
    back-projection, and blending — refiners only do the per-view
    enhancement.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        pass

    @property
    def device(self) -> str:
        """Device the refiner runs on."""
        return getattr(self, "_device", "cuda")

    @device.setter
    def device(self, value: str):
        self._device = value

    @abstractmethod
    def restore(self, image: Image.Image) -> Optional[Image.Image]:
        """Restore a single face image.

        Parameters
        ----------
        image : PIL Image (RGB, uint8 or float)

        Returns
        -------
        restored : PIL Image (RGB, uint8) or None if no face detected
        """
        pass

    def restore_batch(
        self, images: List[Image.Image]
    ) -> List[Optional[Image.Image]]:
        """Restore a batch of images (default: loop over restore())."""
        return [self.restore(img) for img in images]

    def _pil_to_array(self, pil_img: Image.Image) -> np.ndarray:
        """PIL RGB → float32 numpy [0,1]."""
        return np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0

    def _array_to_pil(self, arr: np.ndarray) -> Image.Image:
        """float32 numpy [0,1] → PIL RGB uint8."""
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """(H,W,3) float32 torch tensor [0,1] → PIL RGB uint8."""
        arr = tensor.detach().cpu().numpy()
        if arr.max() > 1.5:
            arr = arr / 255.0
        return self._array_to_pil(arr)

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """PIL RGB → (H,W,3) float32 torch tensor [0,1]."""
        arr = self._pil_to_array(pil_img)
        return torch.from_numpy(arr)

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}')>"
