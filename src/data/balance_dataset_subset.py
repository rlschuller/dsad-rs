import os
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np

from src.utils.data import HdfDataset


def load_labels():
    labels = dict()
    for f in Path("data/processed/metadata/closeup").glob("*.json"):
        with open(str(f), "r") as fd:
            data = json.load(fd)
            for k, v in data.items():
                labels[k] = v["label"]

    return labels


def main(args):
    # set and create output dir
    OUTPUT_DIR = Path("data/processed/datasets")
    if args.personal_folder:
        OUTPUT_DIR = Path("personal/data/processed/datasets")
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    with open(args.input, "r") as fd:
        main_data = json.load(fd)

    print("loading labels..")
    start_t = time.perf_counter()
    labels = load_labels()
    print("finish in ", time.perf_counter() - start_t, " seconds.")

    main_dataset = dict({
        "ones": set(),
        "zeros": set(),
    })
    # for s in ["test"]:
    for s in ["train", "val", "test"]:
        for aid_path in main_data[s]:
            aid = aid_path.split("/")[-1].split(".")[0]
            if labels[aid] == 1:
                main_dataset["ones"].add(aid_path)
            else:
                main_dataset["zeros"].add(aid_path)

    final_class_size = min(len(main_dataset["ones"]), len(main_dataset["zeros"]))

    with open(args.subset, "r") as fd:
        subset_data = json.load(fd)

    if args.X:  # Expand subset
        sub_dataset = dict({
            "train": dict({"ones": set(), "zeros": set()}),
            "val": dict({"ones": set(), "zeros": set()}),
            "test": dict({"ones": set(), "zeros": set()}),
        })
        lens = dict({
            "ones": 0,
            "zeros": 0,
        })
        # for s in ["test"]:
        for s in ["train", "val", "test"]:
            for aid_path in subset_data[s]:
                aid = aid_path.split("/")[-1].split(".")[0]
                if labels[aid] == 1:
                    sub_dataset[s]["ones"].add(aid_path)
                else:
                    sub_dataset[s]["zeros"].add(aid_path)
            lens["ones"] += len(sub_dataset[s]["ones"])
            # remove from main data
            main_dataset["ones"] -= sub_dataset[s]["ones"]

            lens["zeros"] += len(sub_dataset[s]["zeros"])
            main_dataset["zeros"] -= sub_dataset[s]["zeros"]

        main_dataset["ones"] = list(main_dataset["ones"])
        main_dataset["zeros"] = list(main_dataset["zeros"])

        # shuffle and remove elements
        rng = np.random.default_rng()
        # print(f"{final_class_size=}")
        # print(f"{lens=}")
        for k in ["ones", "zeros"]:
            class_size = max(final_class_size - lens[k], 0)
            # print(f"{class_size=}")
            idx = np.array(list(range(len(main_dataset[k]))))
            rng.shuffle(idx)
            d = np.array(list(main_dataset[k]))[idx][:class_size]
            # print("--", len(d))
            main_dataset[k] = d

        # print(">>", len(main_dataset["ones"]), len(main_dataset["zeros"]))
        for c in ["ones", "zeros"]:
            len_k = len(main_dataset[c])
            size_train = math.floor(len_k*subset_data["hparams"]["proportion_train"])
            sub_dataset["train"][c] |= set(main_dataset[c][:size_train])

            size_val = math.floor(len_k*subset_data["hparams"]["proportion_val"])
            sub_dataset["val"][c] |= set(main_dataset[c][size_train:size_train+size_val])
            # print("**", size_train, size_val, len_k - size_train - size_val)

            sub_dataset["test"][c] |= set(main_dataset[c][size_train+size_val:])

    # for s in ["test"]:
    for s in ["train", "val", "test"]:
        # print(s, c, len(sub_dataset[s][c]))
        d = list(sub_dataset[s]["zeros"] | sub_dataset[s]["ones"])
        sorted(d)
        main_data[s] = d

    # fix hparams
    total_len = len(main_data["train"]) + len(main_data["val"]) + len(main_data["test"])
    main_data["hparams"]["proportion_train"] = len(main_data["train"])/total_len
    main_data["hparams"]["proportion_val"] = len(main_data["val"])/total_len
    main_data["hparams"]["name"] += "-balanced"

    base = "-".join(str(Path(args.input).stem).split("-")[:-1])  # remove number at the end
    output = OUTPUT_DIR / f"{base}-balanced_subset-{total_len}.json"
    with open(str(output), "w") as fd:
        json.dump(main_data, fd, indent=4)


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
        "-s",
        "--subset",
        help="Use this json as a subset.",
        default="data/processed/datasets/cloud_filter-closeup-time_series-205899.json"
    )
    parser.add_argument(
        "-X",
        help="Expand subset into a new dataset (keep train/val/test division).",
        action="store_true"
    )
    parser.add_argument(
        "-pf",
        "--personal_folder",
        help="User personal folder for output.",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
