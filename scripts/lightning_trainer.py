from pytorch_lightning.cli import LightningCLI
from src.data.dataset import ChexpertDataModule
from src.model.trainer import ClassifierTrainer


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.classes", "model.classes")
        parser.link_arguments("data.image_size", "model.img_size")
        parser.link_arguments("data.num_channels", "model.num_channels")
        parser.link_arguments(
            "model.hugging_face_model", "data.hugging_face_model"
        )
        parser.link_arguments(
            "model.use_pretrained_model", "data.use_image_net_normalization"
        )


def main():
    cli = MyCLI(
        ClassifierTrainer,
        ChexpertDataModule,
        run=False,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
    cli.trainer.logger.log_hyperparams(cli.config)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
