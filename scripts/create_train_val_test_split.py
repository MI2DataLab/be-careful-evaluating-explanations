# %%
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_CHEXPERT_TRAIN_CSV = "dataset/train_cheXpert.csv"
DEFAULT_SAVE_PATH = "dataset/{}_split.csv"


# %%
def main(args):
    data = pd.read_csv(args.data_path)

    patients = data.Path.str.split("/").apply(lambda x: x[2])
    gss = GroupShuffleSplit(1, test_size=args.val_size + args.test_size)

    train_id, val_test_id = next(gss.split(data, groups=patients))

    train_data = data.iloc[train_id]
    val_test_data = data.iloc[val_test_id].reset_index(drop=True)
    gss = GroupShuffleSplit(1, test_size=args.test_size)
    val_test_patients = val_test_data.Path.str.split("/").apply(lambda x: x[2])
    val_id, test_id = next(gss.split(val_test_data, groups=val_test_patients))
    val_data = val_test_data.iloc[val_id]
    test_data = val_test_data.iloc[test_id]
    if args.verbose:
        print(f"train_data_shape: {train_data.shape}")
        print(f"val_data_shape: {val_data.shape}")
        print(f"test_data_shape: {test_data.shape}")
    train_data.to_csv(args.train_save_path, index=False)
    val_data.to_csv(args.val_save_path, index=False)
    test_data.to_csv(args.test_save_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--data-path", default=DEFAULT_CHEXPERT_TRAIN_CSV
    )
    parser.add_argument(
        "--train-save-path", default=DEFAULT_SAVE_PATH.format("train")
    )
    parser.add_argument(
        "--val-save-path", default=DEFAULT_SAVE_PATH.format("val")
    )
    parser.add_argument(
        "--test-save-path", default=DEFAULT_SAVE_PATH.format("test")
    )
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("-s", "--seed", default=123)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)
