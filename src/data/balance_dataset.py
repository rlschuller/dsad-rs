import os
import json
import math
import argparse
from pathlib import Path

import numpy as np

from src.utils.data import HdfDataset


def balance(name, dataset):
    labels = np.array(list(dataset.labels.values()))
    values, counts = np.unique(labels, return_counts=True)
    print(f"Label counts for '{name}':")
    min_value = 0
    min_count = math.inf
    for v, c in zip(values, counts):
        print(f" {v}: {c}")
        if c < min_count:
            min_value = v
            min_count = c
    print(f"Using class {min_value} as base.")
    final_indices = list()
    rng = np.random.default_rng()
    for v in values:
        # get all indices for this class
        class_indices = np.nonzero(labels == v)[0]
        # shuffle and get the first 'min_count' elements
        rng.shuffle(class_indices)  # in place
        class_indices = class_indices[:min_count]
        # save selected indices to create final dataset
        final_indices.append(class_indices)

    # shuffle to mix the classes
    final_indices = list(np.array(final_indices).flatten())
    sorted(final_indices)

    return np.array(final_indices)


def main(args):
    # set and create output dir
    OUTPUT_DIR = Path("data/processed/datasets")
    if args.personal_folder:
        OUTPUT_DIR = Path("personal/data/processed/datasets")
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    with open(args.input, "r") as fd:
        data = json.load(fd)

    for s in ["train", "val", "test"]:
        # use HDFdataset (lazy)
        ds = HdfDataset(
            dataset_path=args.input,
            subset=s,
            preload_labels=True
        )
        # balance
        indices = balance(s, ds)
        # select only the files in the balanced subset
        files = np.array(data[s])[indices]
        # replace files in data subset
        data[s] = list(files)

    # fix hparams
    total_len = len(data["train"]) + len(data["val"]) + len(data["test"])
    data["hparams"]["proportion_train"] = len(data["train"])/total_len
    data["hparams"]["proportion_val"] = len(data["val"])/total_len
    name = "-".join(args.input.split("/")[-1].split("-")[:-1])
    data["hparams"]["name"] = f"{name}-balanced"

    base = "-".join(str(Path(args.input).stem).split("-")[:-1])  # remove number at the end
    output = OUTPUT_DIR / f"{base}-balanced-{total_len}.json"
    with open(str(output), "w") as fd:
        json.dump(data, fd, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a subset of a dataset respecting label proportions.",
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input dataset json.",
        default="data/processed/datasets/closeup-large-1483982.json",
    )
    parser.add_argument(
        "-pf",
        "--personal_folder",
        help="User personal folder for output.",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
