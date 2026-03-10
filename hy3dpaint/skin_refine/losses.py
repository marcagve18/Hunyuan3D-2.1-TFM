import torch
import torch.nn.functional as F


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic total variation loss for a texture tensor.

    Args:
        x: (1, C, H, W) float tensor.
    Returns:
        Scalar TV loss.
    """
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def gradient_penalty(D, real: torch.Tensor, fake: torch.Tensor,
                     device: str = "cuda") -> torch.Tensor:
    """
    WGAN-GP gradient penalty.

    Args:
        D:    Discriminator module.
        real: (N, 3, H, W) real images in [0, 1].
        fake: (N, 3, H, W) fake images in [0, 1], detached from texture graph.
        device: torch device string.
    Returns:
        Scalar gradient penalty.
    """
    N = real.size(0)
    alpha = torch.rand(N, 1, 1, 1, device=device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    pred_interp = D(interp)
    grads = torch.autograd.grad(
        outputs=pred_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(pred_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gp = ((grads.reshape(N, -1).norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def adversarial_loss_G(D, fakes: torch.Tensor) -> torch.Tensor:
    """Generator adversarial loss (WGAN: minimise -D(fake))."""
    return -D(fakes).mean()


def adversarial_loss_D(D, real: torch.Tensor, fake: torch.Tensor,
                       lambda_gp: float = 10.0,
                       device: str = "cuda") -> torch.Tensor:
    """
    Discriminator WGAN-GP loss.

    Returns:
        loss_D scalar (E[D(fake)] - E[D(real)] + lambda_gp * GP)
    """
    pred_real = D(real)
    pred_fake = D(fake.detach())
    loss = pred_fake.mean() - pred_real.mean()
    gp = gradient_penalty(D, real, fake.detach(), device=device)
    return loss + lambda_gp * gp
