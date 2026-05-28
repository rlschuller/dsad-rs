from pathlib import Path
from math import floor
from tqdm import tqdm
import json


with Path("data/processed/grid.json").open('r') as f:
    grid = json.load(f)

id_coordinates = Path("data/processed/id-coordinates.tsv").open('r').readlines()


id_2_geometry_info = {}

for line in tqdm(id_coordinates):
    id = line.split("\t")[0]
    x = float(line.split("\t")[1])
    y = float(line.split("\t")[2])

    j = floor((x - grid["start"][0])/grid["stride"][0])
    i = floor((y - grid["start"][1])/grid["stride"][1])

    cell = grid["coords_to_name"][str((j, i))]
    #print(f"{id=}\t{x=}\t{y=}\t{i=}\t{j=}\t{cell=}")

    id_2_geometry_info[id] = {
        "x":x,
        "y":y,
        "i":i,
        "j":j,
        "cell":cell,
    }


output_path = "data/processed/id_2_geometry_info.json"

print(f"{output_path=}")
with Path(output_path).open('w') as f:
    json.dump(id_2_geometry_info, f, indent=2)
