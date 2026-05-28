import argparse
import json
import os
import random
from glob import glob
import h5py
from pathlib import Path
from tqdm import tqdm

import numpy as np
from unique_names_generator import get_random_name

parser = argparse.ArgumentParser(
    description=(
        "Combine images-cloud_filter with images-closeup to select alerts without clouds. "
        "Save the resulting ids as two distinct dataset splits: cloud_filter-closeup and cloud_filter-context."
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

args = parser.parse_args()

args.input = "images-closeup"

OUTPUT_DIR = "data/processed/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

list_of_ids =[]
id_2_path = {}
for json_path in tqdm(sorted(list(Path("data/processed/metadata/").glob(f"*cloud_filter*"))), desc="gathering ids"):
    partial_list_of_ids = sorted(json.load(json_path.open("r")).keys()) #h5py.File(hdf_path, "r").keys()
    list_of_ids += partial_list_of_ids

    for id in partial_list_of_ids:
        id_2_path[id] = str(json_path)


list_of_ids = sorted(list_of_ids)
selected_ids = []
list_of_tiffs = [f"data/processed/tiffs/{id}.tif" for id in list_of_ids]
counter = 0

for id in tqdm(list_of_ids, desc="computing l1 norm"):
    cloud_filter_hdf_path = id_2_path[id].replace("metadata", "hdf").replace(".json", ".hdf")
    closeup_hdf_path = cloud_filter_hdf_path.replace("cloud_filter", "closeup")

    closeup_image = np.array(h5py.File(closeup_hdf_path, "r")[id]["input"])
    cloud_filter_image = np.array(h5py.File(cloud_filter_hdf_path, "r")[id]["input"])

    dif_image = closeup_image - cloud_filter_image

    l1_norm = np.linalg.norm(dif_image)
    if l1_norm < 100:
        selected_ids.append(id)

list_of_ids = sorted(selected_ids)
list_of_tiffs = [f"data/processed/tiffs/{id}.tif" for id in list_of_ids]

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

name = "cloud_filter-closeup"
dataset["hparams"]["name"] = name
output_path = f"{OUTPUT_DIR}/{name}-{n_total}.json"
print(f"{output_path=}")
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=4)

name = "cloud_filter-context"
dataset["hparams"]["name"] = name
dataset["hparams"]["input"] = "images-context"

output_path = f"{OUTPUT_DIR}/{name}-{n_total}.json"
print(f"{output_path=}")
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=4)
