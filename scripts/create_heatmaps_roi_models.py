import pickle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import torchvision
import yaml
from captum.attr import LayerGradCam
from tqdm import tqdm

from src.model.trainer import ClassifierWithROITrainer

NEGATIVE_SAVE_FOLDER = "gradcam_maps/negative_regularization"
POSITIVE_SAVE_FOLDER = "gradcam_maps/positive_regularization"


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
    positive_model_checkpoint_path,
    negative_model_checkpoint_path,
    positive_save_folder,
    negative_save_folder,
):
    positive_save_folder = Path(positive_save_folder)
    negative_save_folder = Path(negative_save_folder)
    model_positive_regularization = (
        ClassifierWithROITrainer.load_from_checkpoint(
            positive_model_checkpoint_path
        ).eval()
    )
    model_negative_regularization = (
        ClassifierWithROITrainer.load_from_checkpoint(
            negative_model_checkpoint_path
        ).eval()
    )
    ##
    layer_positive_regularization = (
        model_positive_regularization.classifier.features[
            -2
        ].denselayer16.layers[-1]
    )
    layer_negative_regularization = (
        model_negative_regularization.classifier.features[
            -2
        ].denselayer16.layers[-1]
    )

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
        config["data"],
        pretrained_classes=model_positive_regularization.classes,
    )

    layer_positive_regularization_gc = LayerGradCam(
        model_positive_regularization.classifier, layer_positive_regularization
    )
    layer_negative_regularization_gc = LayerGradCam(
        model_negative_regularization.classifier, layer_negative_regularization
    )

    def calculate_gradcam(
        img,
        attributor_pos,
        attributor_neg,
        class_index,
        cxr_dims,
        label_gt,
        class_name,
    ):
        output_negative = {}
        output_negative["task"] = class_name
        output_negative["cxr_img"] = img
        output_negative["cxr_dims"] = cxr_dims
        output_negative["gt"] = label_gt
        output_positive = {}
        output_positive["task"] = class_name
        output_positive["cxr_img"] = img
        output_positive["cxr_dims"] = cxr_dims
        output_positive["gt"] = label_gt
        attr_pos = attributor_pos.attribute(img, class_index)
        attr_neg = attributor_neg.attribute(img, class_index)
        negative_prob = (
            torch.sigmoid(
                model_negative_regularization.classifier(img)[0, class_index]
            )
            .detach()
            .item()
        )
        positive_prob = (
            torch.sigmoid(
                model_positive_regularization.classifier(img)[0, class_index]
            )
            .detach()
            .item()
        )
        output_positive["map"] = attr_pos
        output_negative["map"] = attr_neg
        output_positive["prob"] = positive_prob
        output_negative["prob"] = negative_prob
        return output_positive, output_negative

    class_name = model_positive_regularization.classes[0]

    positive_save_folder.mkdir(parents=True, exist_ok=True)
    negative_save_folder.mkdir(parents=True, exist_ok=True)

    for img, labels, image_size, img_path in tqdm(
        valid_dataset, total=len(valid_dataset)
    ):
        output_positive, output_negative = calculate_gradcam(
            img.unsqueeze(0),
            layer_positive_regularization_gc,
            layer_negative_regularization_gc,
            model_positive_regularization.selected_class,
            image_size,
            labels[0],
            class_name,
        )
        save_path = "_".join(img_path.split("/")[-3:])
        save_path = save_path.replace(".jpg", f"_{class_name}_map.pkl")
        with open(positive_save_folder / save_path, "wb") as f:
            pickle.dump(output_positive, f)
        with open(negative_save_folder / save_path, "wb") as f:
            pickle.dump(output_negative, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--positive_model_checkpoint_path", type=str, required=True
    )
    parser.add_argument(
        "--negative_model_checkpoint_path", type=str, required=True
    )
    parser.add_argument(
        "--positive_save_folder", type=str, default=POSITIVE_SAVE_FOLDER
    )
    parser.add_argument(
        "--negative_save_folder", type=str, default=NEGATIVE_SAVE_FOLDER
    )
    args = vars(parser.parse_args())
    main(**args)
