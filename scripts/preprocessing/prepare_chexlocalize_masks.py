import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

chexlocalize_path = Path("dataset/chexlocalize/")

with open(chexlocalize_path / "CheXlocalize/gt_annotations_val.json") as f:
    annotations = json.load(f)


def create_mask(polygons, img_dims):
    """
    Creates a binary mask (of the original matrix size) given a list of polygon
        annotations format.
    Args:
        polygons (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]
    Returns:
        mask (np.array): binary mask, 1 where the pixel is predicted to be the,
                                                 pathology, 0 otherwise
    """
    poly = Image.new("1", (img_dims[1], img_dims[0]))
    for polygon in polygons:
        coords = [(point[0], point[1]) for point in polygon]
        ImageDraw.Draw(poly).polygon(coords, outline=1, fill=1)

    binary_mask = np.array(poly, dtype="int")
    return binary_mask


for patient, annotation_dict in tqdm(annotations.items()):
    for key, value in annotation_dict.items():
        if key == "img_size":
            img_size = value
        else:
            mask = create_mask(value, img_size)
            save_path = str(
                chexlocalize_path
                / "CheXpert/val/"
                / patient.split("_")[0]
                / patient.split("_")[1]
                / (
                    "_".join(patient.split("_")[2:])
                    + f"_{key}_mask.npy".replace(" ", "_")
                )
            )
            np.save(save_path, mask, False)

with open(chexlocalize_path / "CheXlocalize/gt_annotations_test.json") as f:
    annotations = json.load(f)

for patient, annotation_dict in tqdm(annotations.items()):
    for key, value in annotation_dict.items():
        if key == "img_size":
            img_size = value
        else:
            mask = create_mask(value, img_size)
            save_path = str(
                chexlocalize_path
                / "CheXpert/test/"
                / patient.split("_")[0]
                / patient.split("_")[1]
                / (
                    "_".join(patient.split("_")[2:])
                    + f"_{key}_mask.npy".replace(" ", "_")
                )
            )
            np.save(save_path, mask, False)
