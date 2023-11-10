from pathlib import Path

import numpy as np
import torch
import torchvision


class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, labels, images_paths, image_size) -> None:

        super().__init__()
        self.images_paths = images_paths
        self.labels = labels
        self.transforms = torchvision.transforms.Resize(image_size)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        img = torchvision.io.read_image(self.images_paths[idx]) / 255.0
        img = self.transforms(img)
        return img, labels


class XRayROIDataset(XRayDataset):
    def __init__(
        self,
        labels,
        images_paths,
        roi_paths,
        image_size,
        inverse_mask: bool = False,
    ):
        super().__init__(labels, images_paths, image_size)
        self.roi_paths = roi_paths
        self.roi_transform = torchvision.transforms.Resize(
            image_size, torchvision.transforms.InterpolationMode.NEAREST
        )
        self.image_size = image_size
        self.inverse_mask = int(inverse_mask)

    def __getitem__(self, idx):
        img, labels = super().__getitem__(idx)
        mask_path = Path(self.roi_paths[idx])
        if mask_path.exists():
            roi_mask = torch.abs(
                torch.tensor(np.load(mask_path), dtype=torch.float32)
                - self.inverse_mask
            )
            roi_mask = self.roi_transform(roi_mask.unsqueeze(0))
        else:
            roi_mask = (
                torch.zeros((1, *self.image_size), dtype=torch.float32) - 1
            )
        return img, labels, roi_mask
