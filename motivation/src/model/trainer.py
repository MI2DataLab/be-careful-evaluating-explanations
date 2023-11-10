from typing import Any, List, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional.classification import accuracy, auroc, f1_score

from src.captum_changes.grad_cam import LayerGradCam

from .densenet import attribution_functions, models_functions


class ClassifierTrainer(LightningModule):
    def __init__(
        self,
        classifier_model: str,
        classes: List[str],
        optimizer_params: dict,
        ignore_hparams: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=ignore_hparams)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.classes = classes
        self.num_classes = len(classes)
        self.metrics_task = "binary" if self.num_classes == 1 else "multilabel"
        self.selected_class = None
        try:
            self.classifier = models_functions[classifier_model.lower()](
                self.num_classes
            )
        except KeyError:
            raise Exception(f"Unsupported model type: {classifier_model}")
        self.metrics = {
            "accuracy": accuracy,
            "AUCROC": auroc,
            "F1": f1_score,
        }

    def forward(self, inputs):
        out = self.classifier(inputs)
        if self.selected_class is not None:
            return out[:, self.selected_class]
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.hparams.optimizer_params
        )

    def _calculate_and_log_metrics(
        self, name: str, preds, labels, loss, on_step=True, on_epoch=True
    ):
        self.log(f"{name}_loss", loss, on_step=on_step, on_epoch=on_epoch)
        for metric_name, metric in self.metrics.items():
            if self.metrics_task == "multilabel":
                metric_val = metric(
                    preds,
                    labels,
                    task=self.metrics_task,
                    num_labels=self.num_classes,
                    average=None,
                )
                for i, class_name in enumerate(self.classes):
                    self.log(
                        f"{name}_{class_name}_{metric_name}",
                        metric_val[i],
                        on_step=on_step,
                        on_epoch=on_epoch,
                    )
            metric_val_macro = metric(
                preds,
                labels,
                task=self.metrics_task,
                num_labels=self.num_classes,
            )
            self.log(
                f"{name}_macro_{metric_name}",
                metric_val_macro,
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def training_step(self, batch, *args):
        image, labels = batch
        if self.selected_class is not None:
            labels = labels[:, self.selected_class]
        output = self(image)
        preds = torch.sigmoid(output)
        loss = self.loss(output, labels)
        self._calculate_and_log_metrics("train", preds, labels, loss)
        return loss

    def validation_step(
        self,
        batch,
        *args: Any,
    ):
        image, labels = batch
        if self.selected_class is not None:
            labels = labels[:, self.selected_class]
        with torch.inference_mode():
            output = self(image)
            preds = torch.sigmoid(output)
            loss = self.loss(output, labels)
            self._calculate_and_log_metrics(
                "val", preds, labels, loss, on_step=False
            )
        return loss


class ClassifierWithROITrainer(ClassifierTrainer):
    def __init__(
        self,
        classifier_model: str,
        classes: List[str],
        optimizer_params: dict,
        location_loss_lambda: int = 1,
        classifier: Optional[torch.nn.Module] = None,
        selected_class: Optional[str] = None,
    ) -> None:
        super().__init__(
            classifier_model,
            classes,
            optimizer_params,
            ignore_hparams=["classifier"] if classifier is not None else None,
        )
        self.location_loss_lambda = location_loss_lambda
        if classifier is not None:
            self.classifier = classifier
        if selected_class is not None:
            self.selected_class = self.classes.index(selected_class)
            self.classes = [classes[self.selected_class]]
            self.num_classes = 1
            self.metrics_task = "binary"
        else:
            self.selected_class = None
        layer = attribution_functions[classifier_model](self.classifier)
        self.attribution = LayerGradCam(self.forward, layer)

    def forward(self, inputs, save_input=False):

        out = self.classifier(inputs)
        if self.selected_class is not None:
            out = out[:, self.selected_class]
        if save_input:
            self.last_outputs = out
        return out

    def training_step(self, batch, *args):
        image, labels, roi_mask = batch
        location_loss = self._get_full_location_loss(image, roi_mask)
        self.log(
            "train_location_loss", location_loss, on_step=True, on_epoch=True
        )
        output = self.last_outputs
        preds = torch.sigmoid(output)
        if self.selected_class is not None:
            labels = labels[:, self.selected_class]
        bce_loss = self.loss(output, labels)
        full_loss = self.location_loss_lambda * location_loss + bce_loss
        self._calculate_and_log_metrics("train", preds, labels, bce_loss)
        self.last_outputs = None
        return full_loss

    def _get_full_location_loss(self, image, attribution_region_mask):
        if self.selected_class is not None:
            attrs = self.attribution.attribute(
                image, additional_forward_args=(True,), retain_graph=True
            )
            return self._calculate_location_loss(
                attrs, attribution_region_mask
            )

        attrs = self.attribution.attribute(
            image, 0, additional_forward_args=(True,), retain_graph=True
        )
        location_loss = self._calculate_location_loss(
            attrs, attribution_region_mask
        )
        for i in range(1, len(self.classes)):
            attrs = self.attribution.attribute(image, i, retain_graph=True)
            location_loss += self._calculate_location_loss(
                attrs, attribution_region_mask
            )
        return location_loss / len(self.classes)

    def _calculate_location_loss(self, attr, attribution_region_mask):
        location_loss = torch.mean(
            torch.pow(
                torch.relu(
                    self.attribution.interpolate(
                        attr, attribution_region_mask.shape[-2:], "bilinear"
                    )
                )
                - attribution_region_mask,
                2,
            ),
            (1, 2, 3),
        )
        location_loss *= ~torch.any(
            torch.flatten(attribution_region_mask < 0, 1), 1
        )

        return location_loss.sum()
