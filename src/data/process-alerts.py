import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Process tif files of a folder, creating corresponding hdf dataset in 'data/processed/hdf/'."
)

name_choices = sorted([p.stem for p in Path("data/raw/images").glob("*")])

parser.add_argument(
    "-n", "--name", choices=name_choices, required=True, help="name of the images to be processed."
)
args = parser.parse_args()



image_name = args.name

for folder in (Path("data/raw/images") / Path(args.name) ).glob("*"):

    relative_output_path = f"{folder.parent.name}/{folder.name}" #.replace("/", "-")

    OUTPUT_FOLDER = "data/processed"

    # if os.path.exists(OUTPUT_FOLDER):
    #    shutil.rmtree(OUTPUT_FOLDER)

    os.makedirs(f"{OUTPUT_FOLDER}/hdf/{image_name}", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/metadata/{image_name}", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/bands/{image_name}", exist_ok=True)

    counter = 0

    id_to_info = {}

    HDF_PATH = f"{OUTPUT_FOLDER}/hdf/{relative_output_path}.hdf"

    BANDS_PATH = f"{OUTPUT_FOLDER}/bands/{relative_output_path}.json"



    if Path(f"{OUTPUT_FOLDER}/metadata/{relative_output_path}.json").is_file() and Path(HDF_PATH).is_file():
        print(f"{relative_output_path} processed already, error..")
        exit(0)

    if Path(HDF_PATH).is_file():
        Path(HDF_PATH).unlink()

    tiff_files = sorted(list(folder.glob("*.tif"))) #list(glob(folder + "/*.tif")))

    if len(tiff_files) == 0:
        print(f"{folder} has 0 tiff_files, skipping...")
        continue


    with next(folder.glob("*bands.json")).open("r")  as f:
        bands = json.load(f)

    with open(BANDS_PATH, "w") as f:
        json.dump(bands, f)

    hdf = h5py.File(HDF_PATH, "w")


    if (folder / "info.json").is_file():
        with (folder / "info.json").open("r") as f:
            info = json.load(f)
    else:
        info = None

    for file in tqdm(tiff_files, desc=relative_output_path, ncols=60):
        id = os.path.basename(file).split(".")[-2]


        extensionless_path = ".".join(str(file).split(".")[:-1])
        label_path = extensionless_path + "-label.json"

        if not os.path.isfile(label_path):
            tqdm.write(f"warning: {label_path} not found. skipping it...")
            continue

        with open(label_path, "r") as f:
            label = int(f.read())

        output_path = f"{OUTPUT_FOLDER}/tiffs/{id}.tif"

        if info:
            id_to_info[id] = dict(info)
        else:
            id_to_info[id] = {}
        id_to_info[id]["label"] = label

        data = np.array(tifffile.imread(file))


        group = hdf.create_group(id)

        group.create_dataset("input", data=data, compression="lzf")
        group["label"] = int(label)

    with open(f"{OUTPUT_FOLDER}/metadata/{relative_output_path}.json", "w") as f:
        json.dump(id_to_info, f)
