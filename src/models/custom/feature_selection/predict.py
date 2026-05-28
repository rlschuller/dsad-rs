import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils.general import get_deterministic_name, get_hostname, get_git_revision, get_date
import sys
from math import exp

from tqdm import trange

parser = argparse.ArgumentParser(
    description = "Save classification predictions as jsonl files."
)

parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="path of the run's folder",
    default="models/logistic_regression/yellow_ermine",
)

args = parser.parse_args()

with (Path(args.path) / "params.json").open("r") as f:
    params = json.load(f)

print(json.dumps(params,indent=2))

selected_paths = params["selected_paths"]


output_folder = Path(args.path) / "predictions"

#output_folder = Path("test") / "predictions"

output_folder.mkdir(exist_ok=True, parents=True)
print(output_folder)

fold_2_predictions = {}


for fold in ["val", "test"]:

    predictions = []

    f = (output_folder / f"{fold}.jsonl").open("w")

    for i, path in enumerate(selected_paths):
        with Path(f"{path}/predictions-without_sigmoid/{fold}.jsonl").open("r") as g:
            predictions.append([json.loads(x) for x in g.readlines()])

    fold_2_predictions[fold] = predictions


    for i in range(len(predictions[0])):
        pred = params["intercept"]

        for j in range(len(selected_paths)):
            pred += params["coef"][j] * predictions[j][i]["pred"]




        pred = 1 / (1 + exp(-pred))

        if (params["coef"][0] * predictions[0][i]["pred"] < 0) and  (params["coef"][1] * predictions[1][i]["pred"] > 0) and ( pred < .5) and (predictions[0][i]["y"] == 0):
            print(i, predictions[0][i], pred)
            for j in range(len(selected_paths)):
                print(params["coef"][j] * predictions[j][i]["pred"])

            if i == 223883:
                exit(0)

        output = predictions[0][i]
        output["pred"] = pred


        f.write(json.dumps(output) + "\n")

    f.close()
    print(len(predictions))
