import argparse
from pathlib import Path
import json
import numpy as np
from src.utils.general import get_deterministic_name, get_hostname, get_git_revision, get_date
import sys
from math import exp

from tqdm import trange

parser = argparse.ArgumentParser(
    description = "Save classification predictions as jsonl files."
)

parser.add_argument(
    "-m",
    "--month",
    type=int,
    default=1,
    help="number from 1 to 12, representing the month",
)

parser.add_argument(
    "-p",
    "--path",
    type=str,
    default="models/residualconv/2024-09-11T17:44:24-03:00-olive_bonobo",
    help="path to the run's folder of the base model",
)

parser.add_argument(
    "--multiply",
    action="store_true",
    help="multiply the characteristic function of the month and the prediction of the model (by default only the indicator function is saved)",
)

parser.add_argument(
    "--tag",
    type=str,
    default="months-imazon_model-2024",
    help="tag to be saved in the hparams file"
)

args = parser.parse_args()

hparams = vars(args)
hparams["model"] = "month"
hparams["month"] = int(hparams["month"])
month = hparams["month"]

assert month >= 1 and month <= 12


if hparams["multiply"]:
    output_folder = Path(f"models/months-multiply/{month:02d}-{hparams['path'].replace('/', '--')}-{args.tag}/")
else:
    output_folder = Path(f"models/months/{month:02d}-{args.tag}/")


predictions_folder = output_folder / "predictions-without_sigmoid"
predictions_folder.mkdir(exist_ok=True, parents=True)


fold_2_predictions = {}

print(hparams["path"])

for fold in ["train", "val", "test"]:

    with (Path(hparams["path"]) / f"predictions-without_sigmoid/{fold}.jsonl").open("r") as f:
        predictions = [json.loads(x) for x in f.readlines()]

    f = (predictions_folder / f"{fold}.jsonl").open("w")


    fold_2_predictions[fold] = predictions


    for i in trange(len(predictions), desc=fold):
        x = predictions[i]

        _month = int(x["path"].split("/")[-1][4:6])

        output = x


        if _month == month:

            if not hparams["multiply"]:
                output["pred"] = 1
        else:
            if hparams["multiply"]:
                output["pred"] = 0
            else:
                output["pred"] = -1


        f.write(json.dumps(output) + "\n")

    f.close()
    print(len(predictions))

with (output_folder / "hparams.json").open("w") as f:
    json.dump(hparams, f, indent=2)

info = {}
with (output_folder / "info.json").open("w") as f:
    date_start = get_date()
    info["hostname"] = get_hostname()
    info["git"] = get_git_revision()
    info["date_start"] = date_start
    info["argv"] = sys.argv
    f.write(json.dumps(info, indent=4)+"\n")
