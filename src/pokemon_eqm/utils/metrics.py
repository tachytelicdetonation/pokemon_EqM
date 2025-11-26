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
            import lpips
        except Exception:
            self.available = False
            self.fid = None
            self.isc = None
            self.lpips_fn = None
            return

        self.available = True
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.isc = InceptionScore(normalize=True).to(device)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
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

    def compute_lpips_diversity(self, samples: torch.Tensor):
        """Compute average pairwise LPIPS distance between samples.

        Args:
            samples: Tensor of shape (N, C, H, W) in range [-1, 1]

        Returns:
            Dictionary with LPIPS diversity metrics
        """
        if not self.available or self.lpips_fn is None:
            return None

        samples = samples.to(self.device)
        n = samples.shape[0]

        if n < 2:
            return {"lpips_diversity": 0.0}

        # Compute pairwise LPIPS distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.lpips_fn(samples[i:i+1], samples[j:j+1])
                distances.append(dist.item())

        mean_dist = sum(distances) / len(distances)

        return {
            "lpips_diversity": mean_dist,
        }
