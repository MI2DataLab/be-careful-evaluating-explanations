from pytorch_lightning.cli import LightningCLI

from src.data.dataset import CheXlocalizeDataModule
from src.model.mask_finetuning import MaskFinetuning


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.selected_class", "data.selected_class")


def main():
    cli = MyCLI(
        MaskFinetuning,
        CheXlocalizeDataModule,
        run=False,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
    cli.trainer.logger.log_hyperparams(cli.config)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
