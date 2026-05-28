import argparse
import json
import os
import random
from glob import glob
import h5py
from pathlib import Path
from tqdm import tqdm

from unique_names_generator import get_random_name

parser = argparse.ArgumentParser(
    description=(
        "Split processed images into train, test and valilation subsets. "
        "Use a glob expression on the processed files (see --input), and save the results as a json file on 'data/processed/datasets/'."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-n",
    "--n",
    default=0,
    help="number of entries in the dataset",
    type=int,
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
    "-i",
    "--input",
    help="input to be used in the glob expression 'data/processed/metadata/*<input>*'",
    default="images-closeup",
)

parser.add_argument(
    "--name",
    type=str,
    help="override dataset name",
)

args = parser.parse_args()

OUTPUT_DIR = "data/processed/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


list_of_ids =[]

for json_path in tqdm(list(Path("data/processed/metadata/").glob(f"{args.input}/*")), desc="gathering ids"):
    list_of_ids += json.load(json_path.open("r")).keys() #h5py.File(hdf_path, "r").keys()
list_of_tiffs = sorted([f"data/processed/tiffs/{id}.tif" for id in list_of_ids])

random.seed(args.seed)
random.shuffle(list_of_tiffs)

if args.n > 0:
    list_of_tiffs = list_of_tiffs[: args.n]

n_total = len(list_of_tiffs)
n_train = int(n_total * args.proportion_train)
n_val = int(n_total * args.proportion_val)
n_test = n_total - n_train - n_val

assert n_train > 0 and n_val > 0 and n_test > 0, f"Invalid split sizes: {n_train=}, {n_val=}, {n_test=}"

dataset = {}

dataset["train"] = sorted(list_of_tiffs[:n_train])
dataset["val"] = sorted(list_of_tiffs[n_train : n_train + n_val])
dataset["test"] = sorted(list_of_tiffs[n_total - n_test :])
dataset["hparams"] = vars(args)

if args.name:
    name = args.name
else:
    random.seed(json.dumps([dataset["test"], dataset["val"], dataset["train"]]))
    name = get_random_name(separator="_", style="lowercase")

output_path = f"{OUTPUT_DIR}/{name}-{n_total}.json"

print(f"{output_path=}")

with open(output_path, "w") as f:
    json.dump(dataset, f, indent=4)
