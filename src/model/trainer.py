from typing import List, Literal, Optional, Tuple

import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torchmetrics.functional.classification import (
    accuracy,
    auroc,
    average_precision,
    f1_score,
)
from transformers import AutoModelForImageClassification

from src.model.base import BaseModel
from src.model.models import (
    SwinForImageClassification,
    VisionTransformer,
    ViTForImageClassification,
    models_functions,
)
from src.model.radimagenet_model import RadImageNetModel


class ClassifierTrainer(BaseModel):
    def __init__(
        self,
        classifier_model: Literal[
            "vit",
            "swin-vit",
            "densenet201",
        ],
        classes: List[str],
        num_channels: int,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        ignore_hparams: Optional[List[str]] = None,
        img_size: Optional[Tuple[int, int]] = None,
        patch_size: Optional[int] = None,
        with_lrp: bool = True,
        hugging_face_model: Optional[str] = None,
        use_pretrained_model: bool = False,
        radimagenet_pretrained_path: Optional[str] = None,
        **_,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=ignore_hparams, logger=False)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.classes = classes
        self.num_classes = len(classes)
        self.metrics_task = "binary" if self.num_classes == 1 else "multilabel"
        self.selected_class = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_channels = num_channels
        if radimagenet_pretrained_path:
            self.classifier = RadImageNetModel.load_from_checkpoint(
                radimagenet_pretrained_path
            ).classifier
            if classifier_model == "densenet201":
                self.classifier.class_layers.out = torch.nn.Linear(
                    self.classifier.class_layers.out.in_features,
                    self.num_classes,
                )
            else:
                self.classifier.classifier = torch.nn.Linear(
                    self.classifier.classifier.in_features, self.num_classes
                )
        else:
            if hugging_face_model:
                self.classifier = (
                        AutoModelForImageClassification.from_pretrained(
                            hugging_face_model,
                            num_labels=self.num_classes,
                            ignore_mismatched_sizes=True,
                        )
                )
            else:
                try:
                    self.classifier = models_functions[
                        classifier_model.lower()
                    ](
                        num_channels=self.num_channels,
                        num_classes=self.num_classes,
                        img_size=img_size,
                        patch_size=patch_size,
                        pretrained=use_pretrained_model,
                    )
                except KeyError as exc:
                    raise Exception(
                        f"Unsupported model type: {classifier_model}"
                    ) from exc
        self.metrics = {
            "accuracy": accuracy,
            "AUCROC": auroc,
            "F1": f1_score,
            "avg_precision": average_precision,
        }

    def set_selected_class(self, selected_class: str) -> None:
        self.selected_class = self.classes.index(selected_class)
        self.classes = [self.classes[self.selected_class]]

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
