from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor


class ChexpertDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        valid_csv: str,
        data_path: str,
        classes: Optional[List[str]],
        image_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = False,
        normalization_coeffs: Tuple[float, float] = (0.5, 0.5),
        batch_size: int = 32,
        dataloader_params: dict = {"num_workers": 8},
        uncertanity_handler: str = "u-zero",
        only_frontal_view: bool = False,
        num_channels: int = 1,
        chexpert_folder_name: str = "CheXpert-v1.0",
        use_image_net_normalization: bool = False,
        hugging_face_model: Optional[str] = None,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.data_path = data_path
        self.classes = classes
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.normalization_coeffs = normalization_coeffs
        self.batch_size = batch_size
        self.dataloader_params = dataloader_params
        self.uncertanity_handler = uncertanity_handler
        self.only_frontal_view = only_frontal_view
        self.num_channels = num_channels
        self.chexpert_folder_name = chexpert_folder_name
        self.use_image_net_normalization = use_image_net_normalization
        self.hugging_face_model = hugging_face_model

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_dataframe = pd.read_csv(self.train_csv).fillna(0)
            train_dataframe.Path = train_dataframe.Path.str.replace(
                "CheXpert-v1.0", self.chexpert_folder_name
            )
            valid_dataframe = pd.read_csv(self.valid_csv).fillna(0)
            valid_dataframe.Path = valid_dataframe.Path.str.replace(
                "CheXpert-v1.0", self.chexpert_folder_name
            )
            if self.only_frontal_view:
                train_dataframe = train_dataframe[
                    train_dataframe.Path.str.contains("frontal")
                ]
                valid_dataframe = valid_dataframe[
                    valid_dataframe.Path.str.contains("frontal")
                ]
            train_path = train_dataframe.Path.apply(
                lambda x: self.data_path + x
            ).values
            valid_path = valid_dataframe.Path.apply(
                lambda x: self.data_path + x
            ).values
            if self.uncertanity_handler == "u-zero":
                train_dataframe = train_dataframe.replace(-1, 0)
                valid_dataframe = valid_dataframe.replace(-1, 0)
            elif self.uncertanity_handler == "u-one":
                train_dataframe = train_dataframe.replace(-1, 1)
                valid_dataframe = valid_dataframe.replace(-1, 1)
            if self.classes:
                classes = self.classes
            else:
                classes = train_dataframe.columns[-14:].tolist()
            train_labels = train_dataframe[classes].values
            valid_labels = valid_dataframe[classes].values

            self.train_dataset = XRayDataset(
                labels=train_labels,
                images_paths=train_path,
                image_size=self.image_size,
                use_augmentation=self.use_augmentation,
                img_normalization_coeffs=self.normalization_coeffs,
                num_channels=self.num_channels,
                use_image_net_normalization=self.use_image_net_normalization,
                hugging_face_model=self.hugging_face_model,
            )
            self.valid_dataset = XRayDataset(
                labels=valid_labels,
                images_paths=valid_path,
                image_size=self.image_size,
                use_augmentation=False,
                img_normalization_coeffs=(
                    self.train_dataset.mean,
                    self.train_dataset.std,
                ),
                num_channels=self.num_channels,
                hugging_face_model=self.hugging_face_model,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )


class CheXlocalizeDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        valid_csv: str,
        data_path: str,
        selected_class: str,
        image_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = False,
        normalization_coeffs: Tuple[float, float] = (0.5, 0.5),
        batch_size: int = 32,
        inverse_mask: bool = False,
        dataloader_params: dict = {"num_workers": 8},
        num_channels: int = 3,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.dataloader_params = dataloader_params
        self.image_size = image_size
        self.normalization_coeffs = normalization_coeffs
        self.num_channels = num_channels
        self.selected_class = selected_class
        self.train_csv = train_csv
        self.inverse_mask = inverse_mask
        self.use_augmentation = use_augmentation
        self.valid_csv = valid_csv

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_csv = pd.read_csv(self.train_csv)
            valid_csv = pd.read_csv(self.valid_csv)
            train_path = train_csv.Path.apply(lambda x: self.data_path + x)
            valid_path = valid_csv.Path.str.replace(
                "CheXpert-v1.0/valid", "val"
            ).apply(lambda x: self.data_path + x)
            train_roi_mask = train_path.apply(
                lambda x: x[:-4]
                + f"_{self.selected_class.replace(' ', '_')}_mask.npy"
            )
            valid_roi_mask = valid_path.apply(
                lambda x: x[:-4]
                + f"_{self.selected_class.replace(' ', '_')}_mask.npy"
            )
            train_labels = train_csv[self.selected_class].values
            valid_labels = valid_csv[self.selected_class].values
            self.train_dataset = XRayROIDataset(
                train_labels,
                train_path,
                train_roi_mask,
                image_size=self.image_size,
                inverse_mask=self.inverse_mask,
                img_normalization_coeffs=self.normalization_coeffs,
                use_augmentation=self.use_augmentation,
                num_channels=self.num_channels,
            )
            self.valid_dataset = XRayROIDataset(
                valid_labels,
                valid_path,
                valid_roi_mask,
                image_size=self.image_size,
                inverse_mask=self.inverse_mask,
                img_normalization_coeffs=self.normalization_coeffs,
                use_augmentation=self.use_augmentation,
                num_channels=self.num_channels,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )


class RadImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        data_path: str,
        classes: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = False,
        normalization_coeffs: Tuple[float, float] = (0.5, 0.5),
        batch_size: int = 32,
        dataloader_params: dict = {"num_workers": 8},
        num_channels: int = 1,
        test_size: float = 0.05,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.classes = classes
        self.csv_path = csv_path
        self.data_path = data_path
        self.dataloader_params = dataloader_params
        self.image_size = image_size
        self.normalization_coeffs = normalization_coeffs
        self.num_channels = num_channels
        self.test_size = test_size
        self.use_augmentation = use_augmentation

    def setup(self, stage: str) -> None:
        if stage == "fit":
            df = pd.read_csv(self.csv_path).fillna(0)
            train_dataframe, valid_dataframe = train_test_split(
                df, test_size=self.test_size
            )
            train_path = train_dataframe.Path.apply(
                lambda x: self.data_path + x
            ).values
            valid_path = valid_dataframe.Path.apply(
                lambda x: self.data_path + x
            ).values
            if self.classes:
                classes = self.classes
            else:
                classes = train_dataframe.columns[3:].tolist()
            train_labels = train_dataframe[classes].values
            train_labels = np.argmax(train_labels, axis=1)
            valid_labels = valid_dataframe[classes].values
            valid_labels = np.argmax(valid_labels, axis=1)

            self.train_dataset = XRayDataset(
                labels=train_labels,
                images_paths=train_path,
                image_size=self.image_size,
                use_augmentation=self.use_augmentation,
                img_normalization_coeffs=self.normalization_coeffs,
                num_channels=self.num_channels,
            )
            self.valid_dataset = XRayDataset(
                labels=valid_labels,
                images_paths=valid_path,
                image_size=self.image_size,
                use_augmentation=False,
                img_normalization_coeffs=(
                    self.train_dataset.mean,
                    self.train_dataset.std,
                ),
                num_channels=self.num_channels,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            **self.dataloader_params,
        )


class XRayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labels,
        images_paths,
        image_size,
        rescale_factor=1 / 255,
        use_augmentation=False,
        img_normalization_coeffs: Tuple[float, float] = (0.5, 0.5),
        rotate_range=15,
        num_channels: int = 1,
        use_image_net_normalization: bool = False,
        hugging_face_model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.images_paths = images_paths
        self.labels = labels
        self.rescale_factor = rescale_factor
        self.num_channels = num_channels
        self.hugging_face_model = hugging_face_model
        if use_image_net_normalization:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif isinstance(img_normalization_coeffs[0], float):
            self.mean = [img_normalization_coeffs[0]] * num_channels
            self.std = [img_normalization_coeffs[1]] * num_channels
        else:
            self.mean, self.std = img_normalization_coeffs

        if hugging_face_model is not None:
            self.transforms = AutoImageProcessor.from_pretrained(
                hugging_face_model
            )
        elif use_augmentation:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(image_size),
                    torchvision.transforms.RandomAffine(degrees=rotate_range),
                    torchvision.transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(image_size),
                    torchvision.transforms.Normalize(self.mean, self.std),
                ]
            )
        if self.num_channels == 3:
            self.read_mode = torchvision.io.ImageReadMode.RGB
        else:
            self.read_mode = torchvision.io.ImageReadMode.GRAY

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        img = (
            torchvision.io.read_image(
                self.images_paths[idx], mode=self.read_mode
            )
            * self.rescale_factor
        )
        if self.hugging_face_model is not None:
            img = self.transforms(img, return_tensors="pt").pixel_values[0]
        else:
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
        rescale_factor=1 / 255,
        use_augmentation=False,
        img_normalization_coeffs: Tuple[float, float] = (0.5, 0.5),
        rotate_range=15,
        num_channels: int = 1,
        transform_roi: bool = True,
    ) -> None:
        super().__init__(
            labels=labels,
            images_paths=images_paths,
            image_size=image_size,
            rescale_factor=rescale_factor,
            use_augmentation=use_augmentation,
            img_normalization_coeffs=img_normalization_coeffs,
            rotate_range=rotate_range,
            num_channels=num_channels,
        )
        self.roi_paths = roi_paths
        self.roi_transform = torchvision.transforms.Resize(
            image_size, torchvision.transforms.InterpolationMode.NEAREST
        )
        self.image_size = image_size
        self.inverse_mask = int(inverse_mask)
        self.transform_roi = transform_roi

    def __getitem__(self, idx):
        img, labels = super().__getitem__(idx)
        mask_path = Path(self.roi_paths[idx])
        if mask_path.exists():
            roi_mask = torch.abs(
                torch.tensor(np.load(mask_path), dtype=torch.float32)
                - self.inverse_mask
            )
            if self.transform_roi:
                roi_mask = self.roi_transform(roi_mask.unsqueeze(0))
        else:
            roi_mask = torch.zeros((1, *self.image_size), dtype=torch.float32)
        return img, labels, roi_mask
