import argparse
from pathlib import Path
import json
import random

parser = argparse.ArgumentParser(
    description="create a balanced split, without alerts that use the old preprocess method (without alerts from 2021/01 to, and including, 2021/08) and alerts without time series."
)

parser.add_argument(
    "-d",
    "--dataset",
    default="data/processed/datasets/time_series-closeup-2769498.json",
    help="original time series split"
)

parser.add_argument(
    "--name",
    default="time_series-closeup-balanced",
)

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=0,
)

args = parser.parse_args()


print(args.dataset)

with Path(args.dataset).open("r") as f:
    dataset = json.load(f)

all_ids = dataset["train"]+dataset["val"]+dataset["test"]
all_ids = [id.split(".")[0].split("/")[-1] for id in all_ids]
print(all_ids[:10])

metadata = {}

for path in Path("data/processed/metadata/closeup/").glob("*.json"):
    with path.open("r") as f:
        metadata.update(json.load(f))

positive_ids = sorted([id for id in all_ids if metadata[id]["label"] == 1])
negative_ids = sorted([id for id in all_ids if metadata[id]["label"] == 0])

assert len(positive_ids) + len(negative_ids) == len(all_ids)

random.seed(args.seed)
random.shuffle(negative_ids)

selected_ids = sorted(positive_ids + negative_ids[:len(positive_ids)])
selected_ids = [f"data/processed/tiffs/{id}.tif" for id in selected_ids]

out_dataset = {}


set_train = set(dataset["train"])
set_val = set(dataset["val"])
set_test = set(dataset["test"])

out_dataset["train"] = []
out_dataset["val"] = []
out_dataset["test"] = []

for id in selected_ids:

    if ("202101-" in id or
        "202102-" in id or
        "202103-" in id or
        "202104-" in id or
        "202105-" in id or
        "202106-" in id or
        "202107-" in id or
        "202108-" in id):
        continue

    if id in set_train:
        out_dataset["train"].append(id)
    if id in set_val:
        out_dataset["val"].append(id)
    if id in set_test:
        out_dataset["test"].append(id)

out_dataset["train"] = sorted(out_dataset["train"])
out_dataset["val"] = sorted(out_dataset["val"])
out_dataset["test"] = sorted(out_dataset["test"])

out_dataset["hparams"] = vars(args)
out_dataset["hparams"]["input"] = "closeup"
size = len(out_dataset['train'])+ len(out_dataset['val']) + len(out_dataset['test'])
with Path(f"data/processed/datasets/{args.name}-{size}.json").open("w") as f:
    json.dump(out_dataset, f, indent=4)
