import monai
from transformers import (
    SwinConfig,
    SwinForImageClassification,
    ViTConfig,
    ViTForImageClassification,
)


def get_densenet121(num_channels: int, num_classes, *args, **kwargs):
    model = monai.networks.nets.DenseNet121(
        spatial_dims=2, in_channels=num_channels, out_channels=num_classes
    )
    return model


def get_densenet264(num_channels: int, num_classes, *args, **kwargs):
    model = monai.networks.nets.DenseNet264(
        spatial_dims=2, in_channels=num_channels, out_channels=num_classes
    )
    return model


def get_densenet201(
    num_channels: int, num_classes, pretrained: bool = False, *args, **kwargs
):
    model = monai.networks.nets.DenseNet201(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=num_classes,
        pretrained=pretrained,
    )
    return model


def get_vit(
    num_channels: int,
    num_classes: int,
    img_size: int = 320,
    patch_size: int = 16,
    with_lrp: bool = True,
    dropout: float = 0.1,
    **kwargs,
):
    if with_lrp:
        model = VisionTransformer(
            img_size=img_size,
            in_chans=num_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    else:
        model = ViTForImageClassification(
            ViTConfig(
                num_labels=num_classes,
                num_channels=num_channels,
                image_size=img_size,
                patch_size=patch_size,
                dropout=dropout,
            )
        )
    return model


def get_swin_vit(
    num_channels: int,
    num_classes: int,
    img_size: int = 320,
    patch_size: int = 4,
    dropout: float = 0.1,
    **kwargs,
):
    model = SwinForImageClassification(
        SwinConfig(
            image_size=img_size,
            num_channels=num_channels,
            num_labels=num_classes,
            patch_size=patch_size,
            dropout=dropout,
        )
    )
    return model


models_functions = {
    "densenet201": get_densenet201,
    "densenet": get_densenet121,
    "densenet264": get_densenet264,
    "swin-vit": get_swin_vit,
    "vit": get_vit,
}
