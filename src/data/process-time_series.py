import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    default="data/raw/time_series/202206/202206-000200000000000017f0.json",
    help="path to a json time series file",
)

parser.add_argument(
    "-of",
    "--output_folder",
    default="data/processed/time_series/",
    help="path to a json time series file",
)

args = parser.parse_args()


path = Path(args.path)

output_path = Path(args.output_folder) / Path(path.parts[-2], path.parts[-1])
output_path.parent.mkdir(parents=True,exist_ok=True)

with path.open("r") as f:
    time_series = json.load(f)

invalid_indexes = []

for k1 in time_series:
    if k1!="dates":
        for k2 in time_series[k1]:
            for i,val in enumerate(time_series[k1][k2]):

                # this test will not discard booleans
                if not isinstance(val, (int, float)):
                    invalid_indexes.append(i)

invalid_indexes = set(invalid_indexes)

processed_time_series = {}

processed_time_series["dates"] = [x for i, x in enumerate(time_series["dates"]) if i not in invalid_indexes]


for k1 in time_series:
    if k1!="dates":
        processed_time_series[k1] = {}
        for k2 in time_series[k1]:
            processed_time_series[k1][k2] = [x for i, x in enumerate(time_series[k1][k2]) if i not in invalid_indexes]


with output_path.open("w") as f:
    json.dump(processed_time_series,f)
