import monai


def get_densenet121(num_classes):
    model = monai.networks.nets.DenseNet121(
        spatial_dims=2, in_channels=1, out_channels=num_classes
    )
    return model


def get_densenet121_attribution_layer(model):
    return model.features[-2].denselayer16.layers[-1]


models_functions = {"densenet": get_densenet121}
attribution_functions = {"densenet": get_densenet121_attribution_layer}
