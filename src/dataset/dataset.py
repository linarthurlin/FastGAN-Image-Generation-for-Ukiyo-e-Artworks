from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ---------------------------------------------------------------------------
# ImageDataset  –  unconditional GAN (FastGAN)
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        train: bool = True,
        crop_scale_min: float = 0.8,
    ):
        super().__init__()
        self.image_size = image_size
        self.train      = train

        base         = datasets.ImageFolder(root=root)
        self.samples = base.samples  # list of (path, class_idx)

        if train:
            self.transform = transforms.Compose([
                # RandomResizedCrop: sample a crop in [crop_scale_min, 1.0] of
                # the image area, then resize to image_size with BICUBIC.
                # Produces position / scale diversity that CenterCrop cannot.
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(crop_scale_min, 1.0),
                    ratio=(0.9, 1.1),          # allow slight aspect-ratio jitter
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                # Mild ColorJitter: preserves the distinctive Ukiyo-e palette
                # while making D harder to overfit on exact colour values.
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                ),
            ])
        else:
            # Eval: deterministic centre crop only
            self.transform = transforms.Compose([
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
            ])

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, _ = self.samples[idx]
        img     = Image.open(path).convert("RGB")
        img     = self.transform(img)
        return self.normalize(self.to_tensor(img))

# ---------------------------------------------------------------------------
# SRDataset  –  super-resolution  (legacy, kept for backward compat)
# ---------------------------------------------------------------------------
class SRDataset(Dataset):
    def __init__(
        self,
        root: str,
        hr_size: int = 128,
        scale_factor: int = 4,
        train: bool = True,
    ):
        super().__init__()
        self.hr_size      = hr_size
        self.lr_size      = hr_size // scale_factor
        self.scale_factor = scale_factor
        self.train        = train

        # Collect all image paths via ImageFolder
        base = datasets.ImageFolder(root=root)
        self.samples = base.samples  # list of (path, class_idx)

        # HR transforms: random crop + optional flip
        hr_tfs = [transforms.RandomCrop(hr_size)]
        if train:
            hr_tfs.append(transforms.RandomHorizontalFlip())
        self.hr_transform = transforms.Compose(hr_tfs)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Ensure image is large enough to crop
        w, h = img.size
        min_side = min(w, h)
        if min_side < self.hr_size:
            scale = self.hr_size / min_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)

        hr = self.hr_transform(img)

        # Bicubic downscale to produce LR
        lr = TF.resize(hr, self.lr_size, interpolation=Image.BICUBIC)

        hr_t = self.normalize(self.to_tensor(hr))
        lr_t = self.normalize(self.to_tensor(lr))

        return lr_t, hr_t


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------
def get_image_dataloader(
    data_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    train: bool = True,
    crop_scale_min: float = 0.8,
) -> DataLoader:
    """Return a DataLoader that yields single real-image batches (FastGAN)."""
    dataset = ImageDataset(
        root=data_dir,
        image_size=image_size,
        train=train,
        crop_scale_min=crop_scale_min,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=train,
    )


def get_dataloader(
    data_dir: str,
    batch_size: int = 16,
    hr_size: int = 128,
    scale_factor: int = 4,
    num_workers: int = 4,
    train: bool = True,
) -> DataLoader:
    """Return a DataLoader that yields (lr, hr) batches (legacy SR)."""
    dataset = SRDataset(
        root=data_dir,
        hr_size=hr_size,
        scale_factor=scale_factor,
        train=train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=train,
    )
    return loader
