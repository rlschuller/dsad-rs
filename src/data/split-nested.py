import argparse
import json
import os
import random
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=(
        "Construct a new dataset by appending new ids to a base dataset."
        "The final result is a larger dataset, in which all of the folds contain the corresponding folds from the base one."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-pt", "--proportion_train", default=0.8, type=float)

parser.add_argument("-pv", "--proportion_val", default=0.1, type=float)

parser.add_argument(
    "-s",
    "--seed",
    default=0,
    type=int,
)

parser.add_argument(
    "-",
    "--input",
    help="input to be used in the glob expression 'data/processed/metadata/*<input>*'",
    default="closeup",
)

parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="data/processed/datasets/2021_22-large-1475501.json",
    help="base dataset",
)

args = parser.parse_args()

OUTPUT_DIR = "data/processed/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


ids =[]

for json_path in tqdm(list(Path("data/processed/metadata/").glob(f"{args.input}/*")), desc="gathering ids"):
    ids += json.load(json_path.open("r")).keys() #h5py.File(hdf_path, "r").keys()

ids = [f"data/processed/tiffs/{id}.tif" for id in ids]

with Path(args.dataset).open("r") as f:
    base_dataset = json.load(f)

old_ids = set(base_dataset["train"]+base_dataset["test"]+base_dataset["val"])

new_ids = sorted([x for x in ids if x not in old_ids])

print(new_ids[:10])
random.seed(args.seed)
random.shuffle(new_ids)
print(new_ids[:10])

n_new = len(new_ids)
n_train = int(n_new * args.proportion_train)
n_val = int(n_new * args.proportion_val)
n_test = n_new - n_train - n_val

assert n_train > 0 and n_val > 0 and n_test > 0, f"Invalid split sizes: {n_train=}, {n_val=}, {n_test=}"

dataset = {}

dataset["train"] = sorted(new_ids[:n_train] + base_dataset["train"])
dataset["val"] = sorted(new_ids[n_train : n_train + n_val] + base_dataset["val"])
dataset["test"] = sorted(new_ids[n_new - n_test :] + base_dataset["test"])
dataset["hparams"] = vars(args)

n_total = n_new + len(old_ids)

output_path = f"{OUTPUT_DIR}/nested-{Path(args.dataset).stem}-{n_total}.json"

print(f"{output_path=}")

with open(output_path, "w") as f:
    json.dump(dataset, f, indent=4)
