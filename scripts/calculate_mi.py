import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

from src.data.dataset import XRayROIDataset
from src.model.trainer import ClassifierTrainer, ClassifierWithROITrainer

with open("config/fine_tuning_model_with_roi.yaml") as f:
    config = yaml.safe_load(f)


def prepare_data(data_config, pretrained_classes, selected_class):

    valid_csv = pd.read_csv(data_config["valid_csv"])

    valid_path = valid_csv.Path.str.replace(
        "CheXpert-v1.0/valid", "val"
    ).apply(lambda x: data_config["data_path"] + x)

    valid_roi_mask = valid_path.apply(
        lambda x: x[:-4] + f"_{selected_class.replace(' ', '_')}_mask.npy"
    )
    valid_labels = valid_csv[pretrained_classes].values
    valid_dataset = XRayROIDataset(
        valid_labels,
        valid_path,
        valid_roi_mask,
        image_size=data_config["image_size"],
        inverse_mask=data_config["inverse_mask"],
    )
    return valid_dataset


def get_predictions(dataset, pos_model, neg_model=None):
    with torch.inference_mode():
        pos_outputs = []
        if neg_model is not None:
            neg_outputs = []
        for real_img, _, _ in tqdm(dataset):
            real_img = real_img.unsqueeze(0)
            pos_outputs.append(
                torch.sigmoid(pos_model(real_img)).detach().numpy()
            )
            if neg_model is not None:
                neg_outputs.append(
                    torch.sigmoid(neg_model(real_img)).detach().numpy()
                )
        if neg_model is None:
            return pos_outputs
        return pos_outputs, neg_outputs


def main(
    positive_regularization_model_Enlarged_Cardiomediastinum_path,
    positive_regularization_model_Atelectasis_path,
    negative_regularization_model_Enlarged_Cardiomediastinum_path,
    negative_regularization_model_Atelectasis_path,
    normal_model_path,
):
    output = {}
    model_positive_regularization = (
        ClassifierWithROITrainer.load_from_checkpoint(
            positive_regularization_model_Enlarged_Cardiomediastinum_path
        ).eval()
    )
    model_negative_regularization = (
        ClassifierWithROITrainer.load_from_checkpoint(
            negative_regularization_model_Enlarged_Cardiomediastinum_path
        ).eval()
    )

    normal_model = ClassifierTrainer.load_from_checkpoint(
        normal_model_path
    ).eval()

    config["data"]["inverse_mask"] = False
    valid_dataset = prepare_data(
        config["data"],
        pretrained_classes=normal_model.classes,
        selected_class=model_positive_regularization.classes[0],
    )

    pos_outputs, neg_outputs = get_predictions(
        valid_dataset,
        model_positive_regularization,
        model_negative_regularization,
    )

    pos_outputs = np.array(pos_outputs)
    neg_outputs = np.array(neg_outputs)

    labels = valid_dataset.labels

    cur_labels = labels[:, model_negative_regularization.selected_class]

    pos_mi, neg_mi = mutual_info_classif(
        pos_outputs, cur_labels
    ), mutual_info_classif(neg_outputs, cur_labels)

    normal_preds = get_predictions(valid_dataset, normal_model)

    normal_preds = np.concatenate(normal_preds, axis=0)

    normal_mi = mutual_info_classif(
        normal_preds[:, model_positive_regularization.selected_class].reshape(
            -1, 1
        ),
        cur_labels,
    )
    output["positive_model_mi_Enlarged_Cardiomediastinum"] = pos_mi[0]
    output["negative_model_mi_Enlarged_Cardiomediastinum"] = neg_mi[0]
    output["normal_model_mi_Enlarged_Cardiomediastinum"] = normal_mi[0]

    model_positive_regularization_Atelectasis = (
        ClassifierWithROITrainer.load_from_checkpoint(
            positive_regularization_model_Atelectasis_path
        ).eval()
    )
    model_negative_regularization_Atelectasis = (
        ClassifierWithROITrainer.load_from_checkpoint(
            negative_regularization_model_Atelectasis_path
        ).eval()
    )

    pos_outputs_Atelectasis, neg_outputs_Atelectasis = get_predictions(
        valid_dataset,
        model_positive_regularization_Atelectasis,
        model_negative_regularization_Atelectasis,
    )

    pos_outputs_Atelectasis = np.array(pos_outputs_Atelectasis)
    neg_outputs_Atelectasis = np.array(neg_outputs_Atelectasis)
    
    cur_labels = labels[
        :, model_negative_regularization_Atelectasis.selected_class
    ]

    pos_mi, neg_mi = mutual_info_classif(
        pos_outputs_Atelectasis, cur_labels
    ), mutual_info_classif(neg_outputs_Atelectasis, cur_labels)

    normal_mi = mutual_info_classif(
        normal_preds[
            :, model_positive_regularization_Atelectasis.selected_class
        ].reshape(-1, 1),
        cur_labels,
    )
    output["positive_model_mi_Atelectasis"] = pos_mi[0]
    output["negative_model_mi_Atelectasis"] = neg_mi[0]
    output["normal_model_mi_Atelectasis"] = normal_mi[0]
    with open("mi_scores.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--positive_regularization_model_Enlarged_Cardiomediastinum_path",
        type=str,
    )
    parser.add_argument(
        "--positive_regularization_model_Atelectasis_path", type=str
    )
    parser.add_argument(
        "--negative_regularization_model_Enlarged_Cardiomediastinum_path",
        type=str,
    )
    parser.add_argument(
        "--negative_regularization_model_Atelectasis_path", type=str
    )
    parser.add_argument("--normal_model_path", type=str)
    args = parser.parse_args()
    main(**vars(args))
