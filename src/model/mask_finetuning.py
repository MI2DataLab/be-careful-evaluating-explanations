from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from src.model.models import (
    SwinForImageClassification,
    VisionTransformer,
    ViTForImageClassification,
)
from src.model.trainer import ClassifierTrainer
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score
from zennit.attribution import Gradient, IntegratedGradients, SmoothGrad
from zennit.composites import EpsilonPlus


class MaskFinetuning(LightningModule):
    def __init__(
        self,
        pretrained_model_checkpoint_path: str,
        selected_class: str,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        xai_method: str = "gradient",
        location_loss_lambda: float = 1.0,
        only_positive_relevance: bool = True,
        epsilon: float = 1e-6,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.location_loss_lambda = location_loss_lambda
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.only_positive_relevance = only_positive_relevance
        self.loss = torch.nn.BCEWithLogitsLoss()
        model = ClassifierTrainer.load_from_checkpoint(
            pretrained_model_checkpoint_path
        )
        self.classifier = model.classifier
        self.num_classes = model.num_classes
        self.prediction_class = selected_class
        self.xai_method = xai_method
        if self.xai_method == "gradient":
            self.xai = Gradient(self, create_graph=True, retain_graph=True)
        elif self.xai_method == "integrated_gradients":
            self.xai = IntegratedGradients(
                self, create_graph=True, retain_graph=True
            )
        elif self.xai_method == "lrp":
            self.xai = Gradient(
                self,
                EpsilonPlus(epsilon),
                create_graph=True,
                retain_graph=True,
            )
        elif self.xai_method == "smoothgrad":
            self.xai = SmoothGrad(self, create_graph=True, retain_graph=True)
        else:
            raise KeyError(f"Unsupported xai method: {self.xai_method}")
        self.selected_class = model.classes.index(selected_class)
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(task="binary"),
                "AUROC": AUROC(task="binary"),
                "F1Score": F1Score(task="binary"),
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
        if self.selected_class is not None:
            return out[:, self.selected_class]
        return out

    def training_step(self, batch, *args):
        image, labels, mask = batch
        output = self(image)
        preds = torch.sigmoid(output)
        loss = self.loss(output, labels)
        location_loss = self.calculate_location_loss(image, labels, mask)
        self.log("location_loss", location_loss, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        full_loss = loss + self.location_loss_lambda * location_loss
        output = self.train_metrics(preds, labels)
        self.log_dict(output, on_step=True)
        return full_loss

    def validation_step(
        self,
        batch,
        batch_idx,
        *args,
    ):
        image, labels, _ = batch
        output = self(image)
        preds = torch.sigmoid(output)
        loss = self.loss(output, labels)
        if batch_idx == 0:
            stacked = torch.stack((preds, labels), dim=1).to(torch.float32)
            self.logger.log_table(
                "valid_preds", columns=["pred", "label"], data=stacked.tolist()
            )
        self.log("valid_loss", loss, on_epoch=True)
        self.valid_metrics.update(preds, labels)
        return loss

    def calculate_location_loss(self, image, labels, mask):
        if labels.sum() == 0:
            return 0
        positive_samples_mask = labels == 1
        image = image[positive_samples_mask]
        mask = mask[positive_samples_mask]
        with self.xai:
            _, relevance = self.xai(image)
        if self.only_positive_relevance:
            relevance = relevance.clamp(min=0)
        relevance = relevance.mean(dim=1)
        relevance = (relevance - relevance.amin(dim=(1, 2), keepdim=True)) / (
            relevance.amax(dim=(1, 2), keepdim=True)
            - relevance.amin(dim=(1, 2), keepdim=True)
            + 1e-6
        )
        location_loss = ((relevance - mask) ** 2).mean(dim=(1, 2))
        location_loss = location_loss.mean()
        return location_loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def on_train_epoch_end(self):
        output = {
            key + "_epoch": value
            for key, value in self.train_metrics.compute().items()
        }
        self.log_dict(output, on_step=False, on_epoch=True)
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True)
        self.valid_metrics.reset()
        return super().on_validation_epoch_end()
