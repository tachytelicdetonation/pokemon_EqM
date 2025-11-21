import torch


def sig_isotropic_gaussian_loss(
    z: torch.Tensor,
    num_projections: int = 64,
    num_freqs: int = 17,
    freq_max: float = 5.0,
    grad_clip: float = 5.0,
    eps: float = 1e-6,
):
    """Characteristic-function SIGReg (CF-SIGReg).

    Projects features to random 1D directions, evaluates the empirical
    characteristic function at a small grid of frequencies, and matches it
    to the N(0,1) characteristic function with an MSE objective. A light
    gradient clamp on the projected features keeps this auxiliary loss from
    destabilizing the main objective.
    """
    if z.dim() > 2:
        z_flat = z.reshape(z.shape[0], -1)
    else:
        z_flat = z

    device = z.device
    proj = torch.randn(num_projections, z_flat.shape[1], device=device)
    proj = proj / (proj.norm(dim=1, keepdim=True).clamp_min(eps))
    projected = z_flat @ proj.t()  # [B, P]

    if grad_clip is not None:
        projected.register_hook(lambda g: g.clamp_(-grad_clip, grad_clip))

    freqs = torch.linspace(-freq_max, freq_max, num_freqs, device=device)
    phase = projected.unsqueeze(-1) * freqs  # [B, P, F]
    window = torch.exp(-0.5 * (freqs / freq_max) ** 2)  # damp high frequencies

    empirical_cf = torch.exp(1j * phase).mean(dim=0)  # [P, F] complex
    gaussian_cf = torch.exp(-0.5 * freqs**2)  # [F] real target

    diff_real = empirical_cf.real - gaussian_cf
    diff_imag = empirical_cf.imag
    cf_mse = (diff_real.pow(2) + diff_imag.pow(2)) * window

    loss = cf_mse.mean()
    return loss, projected.var(dim=0).detach()
