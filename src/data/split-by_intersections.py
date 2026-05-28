import argparse

import json
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset",
    default="data/processed/datasets/time_series-closeup-balanced-225919.json",
    help="base dataset, from which the ids will be filtered",
)

parser.add_argument(
    "-i",
    "--intersection_datasets",
    default=[
        "data/processed/datasets/closeup-copernicus-s1_grd-first_and_last-407041.json",
        "data/processed/datasets/closeup-landsat_8-414648.json",
    ],
    type=str,
    nargs="+",
    help="a list of datasets: only ids that belong to all of the datasets will be selected from the base dataset.",
)

args = parser.parse_args()

hparams = vars(args)

print(hparams)


with Path(hparams["dataset"]).open("r") as f:
    dataset = json.load(f)


set_of_selectable = set(dataset["train"]+dataset["val"]+dataset["test"])


for p in hparams["intersection_datasets"]:
    _set_of_selectable = set()
    with Path(p).open("r") as f:
        _dataset = json.load(f)

    _set_of_selectable.update(_dataset["train"]+_dataset["val"]+_dataset["test"])

    set_of_selectable = set_of_selectable.intersection(_set_of_selectable)

print(len(set_of_selectable))

dataset["train"] = [x for x in dataset["train"] if x in set_of_selectable]
dataset["val"] = [x for x in dataset["val"] if x in set_of_selectable]
dataset["test"] = [x for x in dataset["test"] if x in set_of_selectable]

size = len(dataset["train"] + dataset["val"] + dataset["test"])


output_name = ("--".join(
    ["-".join(hparams["dataset"].split("/")[-1].split("-")[:-1])]
    +
    ["-".join(p.split("/")[-1].split("-")[:-1]) for p in hparams["intersection_datasets"]]
)
    +
    f"-{str(size)}"
)

print(output_name)

output_path = f"data/processed/datasets/{output_name}.json"
print(f"{output_path=}")

hparams["input"] = dataset["hparams"]["input"]

dataset["hparams"] = hparams

with Path(output_path).open("w") as f:
    json.dump(dataset, f, indent=4)

