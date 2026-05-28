import json
import shutil
from pathlib import Path
import argparse
import tifffile
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input",
    default="data/raw/image_collections/closeup-copernicus-s1_grd/"
)

parser.add_argument(
    "-o",
    "--output",
    default="data/raw/images/closeup-copernicus-s1_grd-first_and_last"
)

args = parser.parse_args()

label_paths = [str(x) for x in Path(args.input).glob("**/*label.json")]

label_paths = sorted(label_paths)



alert_folder_2_label_paths = {}

for p in label_paths:
    alert_folder = "/".join(p.split("/")[:-2])

    if alert_folder not in alert_folder_2_label_paths:
        alert_folder_2_label_paths[alert_folder] = []

    alert_folder_2_label_paths[alert_folder].append(p)

alert_folders = sorted(list(alert_folder_2_label_paths.keys()))

#print(label_paths[:10])


for alert_folder in tqdm(alert_folders):
    label_paths = sorted(alert_folder_2_label_paths[alert_folder])

    label_paths_t0 = [x for x in label_paths if "t0" in x.split("/")]
    label_paths_t1 = [x for x in label_paths if "t1" in x.split("/")]

    if len(label_paths_t0) == 0  or len(label_paths_t1) == 0:
        continue

    selected_t0 = label_paths_t0[0]
    selected_t1 = label_paths_t1[-1]

    id = alert_folder.split("/")[-1]
    date = alert_folder.split("/")[-2]

    output_folder= f"{args.output}/{date}/"

    Path(output_folder).mkdir(exist_ok=True, parents=True)

    output_tif = f"{output_folder}/{id}.tif"


    t0 = tifffile.imread(selected_t0.replace("-label.json", ".tif"))
    t1 = tifffile.imread(selected_t1.replace("-label.json", ".tif"))


    min_shape = [min(x, y) for x,y in zip(t0.shape, t1.shape)]

    t0 = t0[:min_shape[0], :min_shape[1], :min_shape[2]]
    t1 = t1[:min_shape[0], :min_shape[1], :min_shape[2]]

    joined = np.concatenate([t0, t1], axis=2)

    bands  = [x+"_t0" for x in json.load(Path(selected_t0.replace("-label.json", "-bands.json")).open("r"))]
    bands += [x+"_t1" for x in json.load(Path(selected_t1.replace("-label.json", "-bands.json")).open("r"))]

    tifffile.imwrite(output_tif, joined)
    shutil.copy(selected_t0, f"{output_folder}/{id}-label.json")

    with Path(f"{output_folder}/{id}-bands.json").open("w") as f:
        json.dump(bands, f)


    #print(f"{min_shape=}")

    #print(f"{selected_t0=}")
    #print(f"{selected_t1=}")
    #print(f"{id=}")
    #print(f"{date=}")
    #print(f"{output_folder=}")
    #print(f"{t0.shape=}")
    #print(f"{t1.shape=}")

    #print(f"{bands=}")

    #print(f"{joined.shape=}")
