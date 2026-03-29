"""skin_refine/refiners/nafnet.py — NAFNet face texture restoration refiner.
=============================================================================
Wraps the trained baby_upscaler NAFNet as a drop-in BaseSkinRefiner.

The orchestrator (SkinTextureRefiner) handles multi-view rendering and UV
baking; this refiner only processes individual rendered views.

Usage (via pipeline)
--------------------
    # Fine-tuned checkpoint (after training)
    pipeline.set_skin_refiner("nafnet")

    # SIDD baseline (before fine-tuning — denoising pretrain only)
    pipeline.set_skin_refiner("nafnet")
    pipeline.config.skin_refine_nafnet_ckpt = None
    pipeline.config.skin_refine_nafnet_sidd_ckpt = "path/to/NAFNet-SIDD-width64.pth"
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..base import BaseSkinRefiner

logger = logging.getLogger(__name__)

# Default path to baby_upscaler repo, relative to this file:
# refiners/ → skin_refine/ → hy3dpaint/ → Hunyuan3D-2.1/ → TFM/ → baby_upscaler/
_DEFAULT_NAFNET_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "baby_upscaler")
)


class NAFNetRefiner(BaseSkinRefiner):
    """NAFNet-based face texture restoration.

    Parameters
    ----------
    ckpt_path : str, optional
        Path to a fine-tuned training checkpoint (e.g. 'runs/phase2/ckpt_best.pth').
        Expected format: {"model": state_dict, ...}.
    sidd_ckpt : str, optional
        Path to the SIDD pretrained checkpoint (e.g. 'ckpts/NAFNet-SIDD-width64.pth').
        Loaded via NAFNet.from_sidd(). Used as the baseline before fine-tuning.
        Ignored if ckpt_path is provided.
    nafnet_root : str, optional
        Path to the baby_upscaler repo root (for importing NAFNet).
        Defaults to the sibling directory of Hunyuan3D-2.1.
    width : int, default=64
    enc_blks : list, default=[2, 2, 4, 8]
    middle_blks : int, default=12
    dec_blks : list, default=[2, 2, 2, 2]
    device : str, default='cuda'
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        sidd_ckpt: Optional[str] = None,
        nafnet_root: Optional[str] = None,
        width: int = 64,
        enc_blks: Optional[list] = None,
        middle_blks: int = 12,
        dec_blks: Optional[list] = None,
        device: str = "cuda",
    ):
        self._device = device

        root = str(Path(nafnet_root or _DEFAULT_NAFNET_ROOT).resolve())
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"[NAFNet] baby_upscaler root not found at '{root}'. "
                "Pass nafnet_root= explicitly."
            )

        if root not in sys.path:
            sys.path.insert(0, root)

        from models.nafnet import NAFNet  # imported from baby_upscaler

        arch_kwargs = dict(
            in_ch       = 3,
            out_ch      = 3,
            width       = width,
            enc_blks    = enc_blks or [2, 2, 4, 8],
            middle_blks = middle_blks,
            dec_blks    = dec_blks or [2, 2, 2, 2],
        )

        if ckpt_path is not None:
            # Fine-tuned checkpoint from training
            self._model = NAFNet(**arch_kwargs).to(device)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self._model.load_state_dict(ckpt["model"])
            n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
            logger.info(f"[NAFNet] Loaded fine-tuned weights ({n_params:.1f}M params) from {ckpt_path}")
        elif sidd_ckpt is not None:
            # SIDD pretrained baseline
            self._model = NAFNet.from_sidd(sidd_ckpt, **arch_kwargs).to(device)
            n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
            logger.info(f"[NAFNet] Loaded SIDD baseline weights ({n_params:.1f}M params) from {sidd_ckpt}")
        else:
            raise ValueError(
                "[NAFNet] Provide either ckpt_path (fine-tuned) or sidd_ckpt (SIDD baseline)."
            )

        self._model.eval()

    @property
    def name(self) -> str:
        return "NAFNet"

    @torch.no_grad()
    def restore(self, image: Image.Image) -> Image.Image:
        """Restore a single rendered face view.

        NAFNet is fully convolutional so any resolution is accepted.

        Parameters
        ----------
        image : PIL Image (RGB)

        Returns
        -------
        PIL Image (RGB) with restored skin texture
        """
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

        out = self._model(x).clamp(0, 1)

        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(out_np)
