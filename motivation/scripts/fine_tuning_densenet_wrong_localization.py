import yaml
from src.model.trainer import (
    ClassifierWithSelectedAttributionRegion,
    ClassifierTrainer,
)
from src.data.dataset import XRayDataset
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

with open("config/wrong_fine_tuning_config.yaml") as f:
    config = yaml.safe_load(f)


def prepare_data(data_config):
    train_csv = pd.read_csv(data_config["train_csv"])
    valid_csv = pd.read_csv(data_config["valid_csv"])
    train_path = train_csv.Path.apply(lambda x: data_config["data_path"] + x)
    valid_path = valid_csv.Path.apply(lambda x: data_config["data_path"] + x)
    classes = train_csv.columns[-14:].tolist()
    train_labels = train_csv[classes].values
    valid_labels = valid_csv[classes].values
    train_dataset = XRayDataset(
        train_labels, train_path, data_config["image_size"]
    )
    valid_dataset = XRayDataset(
        valid_labels, valid_path, data_config["image_size"]
    )
    attribution_region = torch.zeros((data_config["attribution_size"]))
    if data_config["attribution_region"] == "border":
        attribution_region[0] = 1
        attribution_region[-1] = 1
        attribution_region[:, -1] = 1
        attribution_region[:, 0] = 1
    elif data_config["attribution_region"] == "lower_part":
        attribution_region[-2:] = 1
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **data_config["dataloader_params"]
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=False, **data_config["dataloader_params"]
    )
    return train_dataloader, valid_dataloader, classes, attribution_region


def main():
    (
        train_dataloader,
        valid_dataloader,
        classes,
        attribution_region,
    ) = prepare_data(config["data"])
    config["model_config"]["classes"] = classes
    config["model_config"]["attribution_region"] = attribution_region
    model = ClassifierWithSelectedAttributionRegion(
        classifier=ClassifierTrainer.load_from_checkpoint(
            config["pretrained_model_path"]
        ).classifier,
        **config["model_config"]
    )
    logger = WandbLogger(**config["logger_config"])
    callbacks = [ModelCheckpoint(**config["model_checkpoint_callback_config"])]
    trainer = Trainer(
        logger=logger, callbacks=callbacks, **config["trainer_config"]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
