import os
import time
import json
import math
import argparse
from pathlib import Path

import numpy as np

from src.utils.metadata import load_labels


def sample(data, labels, ratio=0.1):
    print("Processing labels...", end="", flush=True )
    start_t = time.perf_counter()
    fold_data = dict()
    for afile in data:
        aid = afile.split("/")[-1].split(".")[0]
        fold_data[aid] = labels[aid]
    elapsed_t = time.perf_counter() - start_t
    print(f"finished in {elapsed_t:0.2f} seconds.")
    fold_labels = list(fold_data.values())
    values, counts = np.unique(fold_labels, return_counts=True)
    print("Input dataset contains:")
    for v, c in zip(values, counts):
        print(f"    Label {v} with {c} instances.")
    rng = np.random.default_rng()
    final_indices = list()
    for v in values:
        # get all indices for this class
        class_indices = np.nonzero(fold_labels == v)[0]
        # shuffle and get the first 'min_count' elements
        rng.shuffle(class_indices)  # in place
        count = math.ceil(ratio*len(class_indices))
        class_indices = class_indices[:count]
        # save selected indices to create final dataset
        final_indices.append(class_indices)
    final_indices = np.hstack(final_indices)
    final_indices = np.sort(final_indices)
    data = np.array(data)
    return data[final_indices]


def main(args):
    print(args)
    # set and create output dir
    OUTPUT_DIR = Path("data/processed/datasets")
    if args.personal_folder:
        OUTPUT_DIR = Path("personal/data/processed/datasets")
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    with open(args.input, "r") as fd:
        data = json.load(fd)

    data_input = data["hparams"]["input"]
    dataset = f"data/processed/metadata/{data_input}"
    print("Loading lables...", end="", flush=True )
    start_t = time.perf_counter()
    labels = load_labels(dataset=dataset)
    elapsed_t = time.perf_counter() - start_t
    print(f"finished in {elapsed_t:0.2f} seconds.")


    for s in ["train", "val", "test"]:
        # apply sample on the data fold
        samples = sample(data[s], labels, ratio=args.proportion)
        # replace files in data subset
        data[s] = list(samples)
    
    # fix hparams
    total_len = len(data["train"]) + len(data["val"]) + len(data["test"])
    data["hparams"]["proportion_train"] = len(data["train"])/total_len
    data["hparams"]["proportion_val"] = len(data["val"])/total_len
    data["hparams"]["name"] += "-balanced"

    base = "-".join(str(Path(args.input).stem).split("-")[:-1])  # remove number at the end
    output = OUTPUT_DIR / f"{base}-sampled-{total_len}.json"
    with open(str(output), "w") as fd:
        json.dump(data, fd, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a subset of a dataset respecting label proportions."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input dataset json.",
        default="data/processed/datasets/closeup-large-1483982.json",
    )
    parser.add_argument(
        "-p",
        "--proportion",
        help="Proportion of the input to save.",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "-pf",
        "--personal_folder",
        help="User personal folder for output.",
        action="store_true"
    )
    args = parser.parse_args()

    main(args)