import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    default="data/label_reviews/cloud_filter-closeup-114030-test.tsv",
    help="path to tsv file"
)

OUTPUT_FOLDER = Path("data/processed/label_reviews")

args = parser.parse_args()
path = Path(args.path)

OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

output_path = (OUTPUT_FOLDER / path.name).with_suffix(".json")

with path.open("r") as f:
    lines = f.readlines()

id_to_label = {}
for line in lines[1:]:
    cols = [x.strip() for x in line.split("\t")]
    id = cols[2]
    label = int(cols[7].lower() == "desmatamento")
    id_to_label[id] = label

with output_path.open("w") as f:
    json.dump(id_to_label, f, indent=4)
