from pytorch_lightning.cli import LightningCLI
from src.data.dataset import RadImageNetDataModule
from src.model.radimagenet_model import RadImageNetModel


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.classes", "model.classes")
        parser.link_arguments("data.image_size", "model.img_size")
        parser.link_arguments("data.num_channels", "model.num_channels")


def main():
    cli = MyCLI(
        RadImageNetModel,
        RadImageNetDataModule,
        run=False,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
    cli.trainer.logger.log_hyperparams(cli.config)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
