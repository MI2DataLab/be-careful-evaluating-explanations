import pickle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import torchvision
import yaml
from captum.attr import LayerGradCam
from tqdm import tqdm

from src.model.trainer import ClassifierTrainer

SAVE_FOLDER = "gradcam_maps/no_regularization"


class GradCamMapDataset(torch.utils.data.Dataset):
    def __init__(self, labels, images_paths, image_size):
        super().__init__()
        self.images_paths = images_paths
        self.labels = labels
        self.transforms = torchvision.transforms.Resize(image_size)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        img = torchvision.io.read_image(self.images_paths[idx]) / 255.0
        image_size = img.shape[-2:]
        img = self.transforms(img)

        return (
            img,
            labels,
            (image_size[1], image_size[0]),
            self.images_paths[idx],
        )


with open("config/fine_tuning_model_with_roi.yaml") as f:
    config = yaml.safe_load(f)


def main(
    model_checkpoint_path: str,
    class_to_generate_heatmaps_for: str,
    save_folder: str,
):
    save_folder = Path(save_folder)
    model = ClassifierTrainer.load_from_checkpoint(model_checkpoint_path).eval()

    layer = model.classifier.features[-2].denselayer16.layers[-1]

    def prepare_data(data_config, pretrained_classes):
        valid_csv = pd.read_csv(data_config["valid_csv"])
        valid_path = valid_csv.Path.str.replace(
            "CheXpert-v1.0/valid", "val"
        ).apply(lambda x: data_config["data_path"] + x)
        valid_labels = valid_csv[pretrained_classes].values
        valid_dataset = GradCamMapDataset(
            valid_labels,
            valid_path,
            image_size=data_config["image_size"],
        )
        return valid_dataset

    valid_dataset = prepare_data(
        config["data"], pretrained_classes=model.classes
    )

    layer_gc = LayerGradCam(model.classifier, layer)

    def calculate_gradcam(
        img, attributor, class_index, class_name, cxr_dims, label_gt
    ):
        class_name = model.classes[class_index]
        output = {}
        output["task"] = class_name
        output["cxr_img"] = img
        output["cxr_dims"] = cxr_dims
        output["gt"] = label_gt
        attr = attributor.attribute(img, class_index)
        prob = (
            torch.sigmoid(model.classifier(img)[0, class_index])
            .detach()
            .item()
        )
        output["map"] = attr
        output["prob"] = prob
        return output

    selected_class = model.classes.index(class_to_generate_heatmaps_for)

    save_folder.mkdir(parents=True, exist_ok=True)

    for img, labels, image_size, img_path in tqdm(
        valid_dataset, total=len(valid_dataset)
    ):
        output = calculate_gradcam(
            img.unsqueeze(0),
            layer_gc,
            selected_class,
            class_to_generate_heatmaps_for,
            image_size,
            labels[0],
        )
        save_path = "_".join(img_path.split("/")[-3:])
        save_path = save_path.replace(
            ".jpg", f"_{class_to_generate_heatmaps_for}_map.pkl"
        )
        with open(save_folder / save_path, "wb") as f:
            pickle.dump(output, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--class_to_generate_heatmaps_for", type=str, required=True
    )
    parser.add_argument("--save_folder", type=str, default=SAVE_FOLDER)
    args = vars(parser.parse_args())
    main(**args)
