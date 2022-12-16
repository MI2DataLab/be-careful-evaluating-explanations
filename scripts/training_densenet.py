import argparse

import pandas as pd
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.data.dataset import XRayDataset
from src.model.trainer import ClassifierTrainer

with open("config/training_config.yaml") as f:
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
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **data_config["dataloader_params"]
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=False, **data_config["dataloader_params"]
    )
    return train_dataloader, valid_dataloader, classes


def main(args):
    train_dataloader, valid_dataloader, classes = prepare_data(config["data"])
    config["model_config"]["classes"] = classes
    model = ClassifierTrainer(**config["model_config"])
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
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    main(args)
