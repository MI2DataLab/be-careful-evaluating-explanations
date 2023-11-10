from pytorch_lightning import LightningModule
import abc
import torch


class BaseModel(LightningModule, abc.ABC):
    classifier: torch.nn.Module
    lr: float

    @abc.abstractclassmethod
    def forward(self, inputs):
        ...

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

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
        *args,
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
