from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from tqdm import tqdm
from zennit.attribution import (
    Attributor,
    Gradient,
    IntegratedGradients,
    SmoothGrad,
)
from zennit.composites import EpsilonPlus

from src.data.dataset import XRayROIDataset
from src.metrics.heatmaps import (
    calculate_relevance_mass_accuracy,
    calculate_relevance_rank_accuracy,
)
from src.model.trainer import ClassifierTrainer


def prepare_data(
    csv_path: str,
    selected_class: str,
    data_path: str,
    image_size: Tuple[int, int] = (224, 224),
    rescale_factor: float = 1 / 255,
    img_normalization_coeffs: Tuple[float, float] = (0.5, 0.5),
    num_channels: int = 3,
) -> Tuple[XRayROIDataset, list]:
    """Prepare data for heatmap generation.

    Args:
        csv_path (str): path to csv file with labels and paths.
        selected_class (str): class to localize.
        data_path (str): path to images folder.
        image_size (int, optional): size of image. Defaults to (224, 224).
        rescale_factor (float, optional): image rescale factor.
            Defaults to 1 / 255.
        img_normalization_coeffs (tuple, optional): mean and std of images.
            Defaults to (0.5, 0.5).
        num_channels (int, optional): number of channels. Defaults to 3.

    Returns:

    """
    data = pd.read_csv(csv_path)
    data = data[data[selected_class] == 1].reset_index(drop=True)
    if data.shape[0] == 0:
        exit()
    path = data.Path.str.replace("CheXpert-v1.0/valid", "val")
    unprocessed_paths = path.tolist()
    path = path.apply(lambda x: data_path + x)
    roi_mask_path = path.apply(
        lambda x: x[:-4] + f"_{selected_class.replace(' ', '_')}_mask.npy"
    )
    labels = data[selected_class].values
    dataset = XRayROIDataset(
        labels=labels,
        images_paths=path.values,
        roi_paths=roi_mask_path.values,
        image_size=image_size,
        rescale_factor=rescale_factor,
        img_normalization_coeffs=img_normalization_coeffs,
        num_channels=num_channels,
        transform_roi=False,
    )
    return dataset, unprocessed_paths


def load_model(
    model_path: str, selected_class: str, device: str
) -> ClassifierTrainer:
    """Loads model and sets selected class.

    Args:
        model_path (str): path to model checkpoint.
        selected_class (str): class to localize.
        device (str): device to load model to.

    Returns:
        ClassifierTrainer: model.
    """
    model = ClassifierTrainer.load_from_checkpoint(model_path, strict=False)
    if model.selected_class is None:
        model.set_selected_class(selected_class)
    model.freeze()
    model.eval()
    model.to(device)
    return model


def prepare_attribution_method(
    model: ClassifierTrainer,
    xai_method: str,
    epsilon: float = 1e-6,
) -> Attributor:
    """Prepares attribution method.

    Args:
        model (ClassifierTrainer): model.
        xai_method (str): attribution method.

    Returns:
        Attributor: attribution method object.
    """
    if xai_method == "gradient":
        xai = Gradient(model)
    elif xai_method == "integrated_gradients":
        xai = IntegratedGradients(model)
    elif xai_method == "lrp":
        xai = Gradient(model, EpsilonPlus(epsilon=epsilon))
    elif xai_method == "smoothgrad":
        xai = SmoothGrad(model)
    return xai


def main():
    parser = ArgumentParser()
    parser.add_function_arguments(prepare_data, "data_config")
    parser.add_function_arguments(load_model, "model_config", skip=["model"])
    parser.add_function_arguments(
        prepare_attribution_method, "xai_config", skip=["model"]
    )
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--selected_class", type=str, required=True)
    parser.add_argument("--xai_method", type=str, required=True)
    parser.link_arguments(
        "selected_class", "data_config.selected_class", apply_on="parse"
    )
    parser.link_arguments(
        "selected_class", "model_config.selected_class", apply_on="parse"
    )
    parser.link_arguments(
        "xai_method", "xai_config.xai_method", apply_on="parse"
    )
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    cfg = parser.parse_args()
    selected_class = cfg.selected_class
    xai_method = cfg.xai_method
    save_folder = Path(cfg.save_folder)
    dataset, unprocessed_paths = prepare_data(**cfg.data_config.as_dict())
    model = load_model(**cfg.model_config.as_dict())
    xai = prepare_attribution_method(model, **cfg.xai_config.as_dict())
    outputs = []
    for i, (image, _, roi) in tqdm(
        enumerate(dataset), desc="Generating heatmaps", total=len(dataset)
    ):
        if not Path(dataset.roi_paths[i]).exists():
            continue
        save_path = save_folder / unprocessed_paths[i].replace(
            ".jpg", f"{selected_class}_{xai_method}_relevance.npy"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        image = image.unsqueeze(0)
        roi = roi.unsqueeze(0)
        with xai:
            output, relevance = xai(image)
        output = torch.sigmoid(output)
        with open(save_path, "wb") as f:
            np.save(f, relevance.cpu().numpy())
        relevance = torch.nn.functional.interpolate(
            relevance, roi.size()[1:], mode="bilinear"
        )[0]
        mass_accuracy = calculate_relevance_mass_accuracy(
            relevance, roi, relevance_pooling_type="pos_sum"
        )
        rank_accuracy = calculate_relevance_rank_accuracy(
            relevance, roi, relevance_pooling_type="pos_sum"
        )
        outputs.append(
            (
                unprocessed_paths[i],
                output.cpu().detach().numpy()[0],
                mass_accuracy,
                rank_accuracy,
            )
        )
    output_df = pd.DataFrame(
        outputs, columns=["path", "output", "mass_accuracy", "rank_accuracy"]
    )
    output_df.to_csv(
        f"{save_folder}/_{selected_class}_{xai_method}" "_output.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
