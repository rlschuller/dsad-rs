import argparse
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--input_dataset",
    default="data/processed/datasets/time_series-closeup-balanced-225919.json"
)

parser.add_argument(
    "-y",
    "--year_for_test_and_val",
    help="year not included in train",
    type=int,
    default=2022,
)

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=0,
)


args = parser.parse_args()


input_dataset = json.load(Path(args.input_dataset).open("r"))
all_ids = sorted(input_dataset["train"] + input_dataset["val"] + input_dataset["test"])

output_path = Path(f"data/processed/datasets/balanced-split_by_year-{len(all_ids)}.json")

dataset = {}

train_ids = []
other_ids = []
for id in all_ids:
    if str(args.year_for_test_and_val) != id.split("/")[-1][:4]:
        train_ids.append(id)
    else:
        other_ids.append(id)


print(all_ids[:10])

print(f"{len(all_ids)=}")
print(f"{len(train_ids)=}")

dataset["train"] = train_ids

random.seed(0)
random.shuffle(other_ids)

dataset["val"] = sorted(other_ids[:int(len(other_ids)/2)])
dataset["test"] = sorted(other_ids[int(len(other_ids)/2):])

dataset["hparams"] = vars(args)
dataset["hparams"]["input"] = "closeup"
print(f"{output_path=}")
with output_path.open("w") as f:
    json.dump(dataset, f, indent=4)
