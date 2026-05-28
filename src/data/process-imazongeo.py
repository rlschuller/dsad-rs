from pathlib import Path
import sys
from subprocess import check_output
from subprocess import Popen as popen
import numpy as np
folders = list(str(p.parent) for p in Path("data/raw/imazongeo/").glob("**/*.shx"))
import json
from tqdm import tqdm
#for f in folders:
#    date = f.split("/")[3]
#    #data = check_output(["ogrinfo", "-al", "-features", "-json", f]).decode(sys.stdout.encoding)
#    out_folder = Path(f"data/processed/imazongeo/{date}")
#    out_folder.mkdir(parents=True, exist_ok=True)
#
#    # convert coordinates to WGS84 (and fix winding order)
#    check_output(["ogr2ogr", "-t_srs", "WGS84", str(out_folder), f])
#
#    # compute json
#    data = check_output(["ogrinfo", "-al", "-features", "-json", str(out_folder)]).decode(sys.stdout.encoding)
#
#    with Path(f"data/processed/imazongeo/{date}.json").open("w") as g:
#        g.write(data)
#

jsons = list(Path("data/processed/imazongeo/").glob("*.json"))

tsv_header = "x\ty\tpred\tlabel\tid\n"


f_out = Path("data/processed/imazongeo.tsv").open("w")

f_out.write(tsv_header)

from collections.abc import Iterable

def flatten(v):
    if isinstance(v, Iterable):
        out = []
        for x in v:
            out.extend(flatten(x))
        return out
    else:
        return [v]

weird = 0
for p in tqdm(jsons):
    date = str(p).split("/")[-1].split(".")[0]
    with p.open("r") as f:
        data = json.load(f)

    for i in range(len(data["layers"][0]["features"])):
        properties = data["layers"][0]["features"][i]["properties"]
        coord = flatten(data["layers"][0]["features"][i]["geometry"]["coordinates"])

        if coord[2] != 0.0:

            try:
                coord = np.array(coord).reshape(-1,2)
                coord = np.mean(np.array(coord), axis=0)
                #coord = np.mean(np.array(sum(data["layers"][0]["features"][i]["geometry"]["coordinates"])), axis=0)
            except:

                print()
                print()
                coord = flatten(data["layers"][0]["features"][i]["geometry"]["coordinates"])
                print(coord)
                print()
                #print(data["layers"][0]["features"][i]["geometry"]["coordinates"])
                exit(1)

            #if "AREA" in properties:
            #    area = properties["AREA"]
            #elif "Area" in properties:
            #    area = properties["Area"]
            #elif "F_AREA" in properties:
            #    area = properties["F_AREA"]
            #elif "area" in properties:
            #    area = properties["area"]
            #elif "Shape_Area" in properties:
            #    area = properties["Shape_Area"]
            #elif "area_km2" in properties:
            #    area = properties["area_km2"]
            #else:
            #    print("area not found")
            #    exit(1)

            tsv_line = f"{coord[0]}\t{coord[1]}\t1.0\t1.0\t{date}-{i:016d}\n"

            f_out.write(tsv_line)
        else:
            weird += 1

print(f"{weird=}")

    #month = data["layers"][0]["features"][0]["properties"]["mes"]
    #year = data["layers"][0]["features"][0]["properties"]["ano"]

f_out.close()
