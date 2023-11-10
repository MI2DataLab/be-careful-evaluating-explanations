from typing import List, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score

from src.model.models import (
    SwinForImageClassification,
    VisionTransformer,
    ViTForImageClassification,
    models_functions,
)


class RadImageNetModel(pl.LightningModule):
    def __init__(
        self,
        classifier_model: Literal[
            "densenet",
            "vit",
            "swin-vit",
            "convnext",
            "densenet201",
            "densenet264",
        ],
        classes: List[str],
        num_channels: int,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        img_size: Optional[Tuple[int, int]] = None,
        patch_size: Optional[int] = None,
        dropout: float = 0.1,
        **_,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.loss = torch.nn.CrossEntropyLoss()
        self.classes = classes
        self.num_classes = len(classes)
        self.metrics_task = "binary" if self.num_classes == 1 else "multiclass"
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_channels = num_channels

        try:
            self.classifier = models_functions[classifier_model.lower()](
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                img_size=img_size,
                patch_size=patch_size,
                with_lrp=False,
                dropout=dropout,
            )
        except KeyError as exc:
            raise KeyError(
                f"Unsupported model type: {classifier_model}"
            ) from exc
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(
                    self.metrics_task, num_classes=self.num_classes
                ),
                "Top5_Accuracy": Accuracy(
                    task=self.metrics_task,
                    top_k=5,
                    num_classes=self.num_classes,
                    name="top5_Accuracy",
                ),
                "AUROC": AUROC(
                    self.metrics_task, num_classes=self.num_classes
                ),
                "F1Score": F1Score(
                    self.metrics_task, num_classes=self.num_classes
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, inputs, *args):
        if isinstance(self.classifier, VisionTransformer):
            out = self.classifier(inputs, xai_inference=False)
        else:
            out = self.classifier(inputs)
        if isinstance(
            self.classifier,
            (
                SwinForImageClassification,
                ViTForImageClassification,
            ),
        ):
            out = out.logits
        return out

    def training_step(self, batch, *args):
        image, labels = batch
        output = self(image)
        preds = torch.softmax(output, dim=-1)
        loss = self.loss(output, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        output = self.train_metrics(preds, labels)
        self.log_dict(output, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch,
        *args,
    ):
        image, labels = batch
        output = self(image)
        preds = torch.softmax(output, dim=-1)
        loss = self.loss(output, labels)
        self.log("valid_loss", loss, on_epoch=True)
        self.valid_metrics.update(preds, labels)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True)
        self.valid_metrics.reset()
        self.train_metrics.reset()
        return super().on_validation_epoch_end()
