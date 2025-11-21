import torch


class MetricsHelper:
    """Lightweight wrapper around torchmetrics for FID / Inception Score.

    If torchmetrics is not installed, `available` will be False and compute()
    will return None.
    """

    def __init__(self, device):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.inception import InceptionScore
        except Exception:
            self.available = False
            self.fid = None
            self.isc = None
            return

        self.available = True
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.isc = InceptionScore(normalize=True).to(device)
        self.device = device

    def compute(self, real: torch.Tensor, fake: torch.Tensor):
        if not self.available:
            return None
        real = real.to(self.device)
        fake = fake.to(self.device)
        self.fid.reset(); self.isc.reset()
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        fid_score = self.fid.compute().item()
        self.isc.update(fake)
        is_mean, is_std = self.isc.compute()
        return {
            "fid": fid_score,
            "is_mean": is_mean.item(),
            "is_std": is_std.item(),
        }
