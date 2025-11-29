"""
Multi-View Augmentation for LejEPA Integration.

Implements the multi-view augmentation strategy from LejEPA:
- Global views: Large crops covering 30-100% of the image
- Local views: Small crops covering 5-30% of the image

Each view receives identical color and geometric transforms.

Reference: LejEPA paper - arXiv:2511.08544
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from typing import List, Tuple, Optional
import random
import math


class GaussianBlur:
    """Gaussian blur with random sigma."""
    def __init__(self, p: float = 0.5, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            kernel_size = int(2 * round(3 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        return img


class Solarization:
    """Solarization with random threshold."""
    def __init__(self, p: float = 0.2, threshold: float = 0.5):
        self.p = p
        self.threshold = threshold

    def __call__(self, img):
        if random.random() < self.p:
            return TF.solarize(img, self.threshold)
        return img


class MultiViewTransform:
    """
    Multi-View Transform for LejEPA-style training.

    Creates multiple augmented views of the same image:
    - Global views: Large context crops
    - Local views: Small detail crops

    All views share the same color transform but different spatial crops.
    """
    def __init__(
        self,
        image_size: int = 256,
        num_global_views: int = 2,
        num_local_views: int = 6,
        global_scale: Tuple[float, float] = (0.4, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.4),
        global_size: Optional[int] = None,
        local_size: Optional[int] = None,
        # Color augmentation
        color_jitter: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1),
        grayscale_p: float = 0.2,
        blur_p: float = 0.5,
        solarize_p: float = 0.0,  # Usually 0 for global, >0 for local
    ):
        """
        Args:
            image_size: Base image size
            num_global_views: Number of global (large context) views
            num_local_views: Number of local (detail) views
            global_scale: Scale range for global crops (fraction of image)
            local_scale: Scale range for local crops (fraction of image)
            global_size: Output size for global views (default: image_size)
            local_size: Output size for local views (default: image_size // 2)
            color_jitter: (brightness, contrast, saturation, hue)
            grayscale_p: Probability of converting to grayscale
            blur_p: Probability of Gaussian blur
            solarize_p: Probability of solarization
        """
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views

        global_size = global_size or image_size
        local_size = local_size or image_size // 2

        # Shared color transform
        self.color_transform = transforms.Compose([
            transforms.ColorJitter(*color_jitter),
            transforms.RandomGrayscale(p=grayscale_p),
        ])

        # Global view transforms
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_size,
                scale=global_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            self.color_transform,
            GaussianBlur(p=blur_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Local view transforms (different scale, may include solarization)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=local_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            self.color_transform,
            GaussianBlur(p=blur_p),
            Solarization(p=solarize_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, img) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate multiple views of the input image.

        Args:
            img: PIL Image

        Returns:
            (global_views, local_views) lists of tensors
        """
        global_views = [self.global_transform(img) for _ in range(self.num_global_views)]
        local_views = [self.local_transform(img) for _ in range(self.num_local_views)]

        return global_views, local_views


class LejEPADataTransform:
    """
    Complete data transform for LejEPA training in EqM context.

    For generative models, we may want:
    1. A clean view (for the main EqM loss)
    2. Multiple augmented views (for LejEPA losses)

    This transform returns both.
    """
    def __init__(
        self,
        image_size: int = 256,
        use_multi_view: bool = True,
        num_views: int = 4,  # Total augmented views (for simplicity)
        view_scale: Tuple[float, float] = (0.3, 1.0),
        # Standard augmentation
        color_jitter_strength: float = 0.4,
        grayscale_p: float = 0.2,
        blur_p: float = 0.5,
        horizontal_flip_p: float = 0.5,
    ):
        """
        Args:
            image_size: Target image size
            use_multi_view: Whether to generate multiple views
            num_views: Number of augmented views (if use_multi_view)
            view_scale: Scale range for random crops
            color_jitter_strength: Strength of color augmentation
            grayscale_p: Grayscale probability
            blur_p: Blur probability
            horizontal_flip_p: Horizontal flip probability
        """
        self.image_size = image_size
        self.use_multi_view = use_multi_view
        self.num_views = num_views

        # Base transform (clean, minimal augmentation)
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=horizontal_flip_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Augmented view transform
        if use_multi_view:
            self.aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=view_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=color_jitter_strength,
                    contrast=color_jitter_strength,
                    saturation=color_jitter_strength,
                    hue=color_jitter_strength * 0.25,
                ),
                transforms.RandomGrayscale(p=grayscale_p),
                GaussianBlur(p=blur_p),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __call__(self, img) -> dict:
        """
        Transform input image.

        Args:
            img: PIL Image

        Returns:
            Dict with 'base' tensor and optionally 'views' list
        """
        result = {
            'base': self.base_transform(img),
        }

        if self.use_multi_view:
            result['views'] = [self.aug_transform(img) for _ in range(self.num_views)]

        return result


class LatentMultiViewAugment(nn.Module):
    """
    Multi-View Augmentation in Latent Space.

    For efficiency, we can also create views by augmenting in latent space.
    This is useful when the VAE encoding is expensive.

    Supported augmentations:
    - Spatial crops (in latent space)
    - Feature noise (Gaussian)
    - Feature dropout
    """
    def __init__(
        self,
        num_views: int = 4,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        noise_std: float = 0.1,
        dropout_p: float = 0.1,
    ):
        """
        Args:
            num_views: Number of augmented views to generate
            crop_scale: Scale range for spatial crops in latent space
            noise_std: Standard deviation of Gaussian noise
            dropout_p: Feature dropout probability
        """
        super().__init__()
        self.num_views = num_views
        self.crop_scale = crop_scale
        self.noise_std = noise_std
        self.dropout_p = dropout_p

    def random_crop(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Random spatial crop in latent space."""
        B, C, H, W = x.shape
        new_h = int(H * scale)
        new_w = int(W * scale)

        if new_h >= H and new_w >= W:
            return x

        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)

        cropped = x[:, :, top:top+new_h, left:left+new_w]

        # Resize back to original size
        cropped = torch.nn.functional.interpolate(
            cropped, size=(H, W), mode='bilinear', align_corners=False
        )

        return cropped

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate augmented views in latent space.

        Args:
            z: Latent tensor [B, C, H, W]

        Returns:
            List of augmented latent tensors
        """
        views = []

        for _ in range(self.num_views):
            z_aug = z.clone()

            # Random spatial crop
            scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
            z_aug = self.random_crop(z_aug, scale)

            # Add Gaussian noise
            if self.noise_std > 0 and self.training:
                noise = torch.randn_like(z_aug) * self.noise_std
                z_aug = z_aug + noise

            # Feature dropout
            if self.dropout_p > 0 and self.training:
                mask = torch.rand_like(z_aug) > self.dropout_p
                z_aug = z_aug * mask.float()

            views.append(z_aug)

        return views


def create_lejepa_transforms(
    image_size: int = 256,
    mode: str = 'full',
    num_views: int = 4,
) -> transforms.Compose:
    """
    Factory function to create LejEPA-compatible transforms.

    Args:
        image_size: Target image size
        mode: 'minimal', 'standard', or 'full'
        num_views: Number of views for multi-view mode

    Returns:
        Transform object
    """
    if mode == 'minimal':
        # Just basic preprocessing
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif mode == 'standard':
        # Standard training augmentation
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif mode == 'full':
        # Full LejEPA multi-view transform
        return LejEPADataTransform(
            image_size=image_size,
            use_multi_view=True,
            num_views=num_views,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
