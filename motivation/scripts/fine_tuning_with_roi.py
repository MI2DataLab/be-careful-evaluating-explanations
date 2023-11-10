import argparse

import pandas as pd
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.data.dataset import XRayDataset, XRayROIDataset
from src.model.trainer import ClassifierTrainer, ClassifierWithROITrainer

with open("config/fine_tuning_model_with_roi.yaml") as f:
    config = yaml.safe_load(f)


def prepare_data(data_config, pretrained_classes, selected_class):
    train_csv = pd.read_csv(data_config["train_csv"])
    valid_csv = pd.read_csv(data_config["valid_csv"])
    train_path = train_csv.Path.apply(lambda x: data_config["data_path"] + x)
    valid_path = valid_csv.Path.str.replace(
        "CheXpert-v1.0/valid", "val"
    ).apply(lambda x: data_config["data_path"] + x)
    train_roi_mask = train_path.apply(
        lambda x: x[:-4] + f"_{selected_class.replace(' ', '_')}_mask.npy"
    )
    train_labels = train_csv[pretrained_classes].values
    valid_labels = valid_csv[pretrained_classes].values
    train_dataset = XRayROIDataset(
        train_labels,
        train_path,
        train_roi_mask,
        image_size=data_config["image_size"],
        inverse_mask=data_config["inverse_mask"],
    )
    valid_dataset = XRayDataset(
        valid_labels, valid_path, image_size=data_config["image_size"]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **data_config["dataloader_params"]
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=False, **data_config["dataloader_params"]
    )
    return train_dataloader, valid_dataloader


def main(args):
    pretrained = ClassifierTrainer.load_from_checkpoint(
        config["pretrained_model_path"]
    )
    config["data"]["inverse_mask"] = args.inverse_roi_mask
    (train_dataloader, valid_dataloader,) = prepare_data(
        config["data"],
        pretrained_classes=pretrained.classes,
        selected_class=config["model_config"]["selected_class"],
    )

    config["model_config"]["classes"] = pretrained.classes
    model = ClassifierWithROITrainer(
        classifier=pretrained.classifier, **config["model_config"]
    )
    if args.use_wandb:
        logger = WandbLogger(**config["logger_config"])
    else:
        logger = None
    callbacks = [ModelCheckpoint(**config["model_checkpoint_callback_config"])]
    trainer = Trainer(
        logger=logger, callbacks=callbacks, **config["trainer_config"]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverse_roi_mask", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    main(args)
