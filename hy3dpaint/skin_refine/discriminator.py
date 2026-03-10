import pickle
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# stylegan2-ada-pytorch must be on the path for pickle to deserialise their
# custom classes.  We look for it relative to this file's repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SG2_PATH = os.path.join(_REPO_ROOT, "stylegan2-ada-pytorch")
if os.path.isdir(_SG2_PATH) and _SG2_PATH not in sys.path:
    sys.path.insert(0, _SG2_PATH)


class StyleGAN2Discriminator(nn.Module):
    """
    Loads pretrained StyleGAN2 discriminator from FFHQ checkpoint.
    Wraps it for use in texture optimization.

    The checkpoint is the standard stylegan2-ada-pytorch .pkl file containing
    keys 'G', 'G_ema', 'D', 'training_set_kwargs', etc.

    The StyleGAN2 discriminator expects inputs in [-1, 1] at 1024x1024.
    We rescale and upsample internally so callers can pass [0, 1] images
    at any resolution.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)

        self.D = data["D"].to(device)
        self.D.train()  # keep in train mode so gradients flow for fine-tuning

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W) float tensor with values in [0, 1].
        Returns:
            Scalar logit tensor of shape (N, 1) or (N,) depending on D head.
        """
        # Rescale from [0, 1] → [-1, 1] as expected by StyleGAN2 D
        x = x * 2.0 - 1.0

        # Upsample to 1024x1024 if needed
        if x.shape[-1] != 1024:
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

        # StyleGAN2 D second arg is class label; None = unconditional
        return self.D(x, None)
