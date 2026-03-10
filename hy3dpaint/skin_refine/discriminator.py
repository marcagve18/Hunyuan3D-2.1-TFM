import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Point the CUDA extension cache to a fast local path before StyleGAN2 imports
# its ops (upfirdn2d_plugin, bias_act_plugin).  Without this the default cache
# lands on a slow shared filesystem and the compilation appears to hang.
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions_sg2"),
)

# stylegan2-ada-pytorch must be on sys.path for pickle to deserialise its
# custom classes.  Locate it relative to this file's repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SG2_PATH  = os.path.join(_REPO_ROOT, "stylegan2-ada-pytorch")
if os.path.isdir(_SG2_PATH) and _SG2_PATH not in sys.path:
    sys.path.insert(0, _SG2_PATH)


class StyleGAN2Discriminator(nn.Module):
    """
    Pretrained StyleGAN2 FFHQ discriminator, frozen, used as a fixed
    realism critic for texture optimisation.

    Input : (N, 3, H, W) float [0, 1], any resolution.
    Output: (N, 1) logit — higher means "more like a real face".
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        print(f"[StyleGAN2D] Loading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        self.D = data["D"].to(device)
        self.D.eval()
        for p in self.D.parameters():
            p.requires_grad_(False)
        print("[StyleGAN2D] Discriminator ready (frozen).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [0, 1] → [-1, 1]
        x = x * 2.0 - 1.0
        if x.shape[-1] != 1024:
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        return self.D(x, None)
