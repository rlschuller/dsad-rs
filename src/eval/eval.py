import argparse
import json
import os
from glob import glob

from src.utils.log import get_log_info
from src.utils.general import run_notebook_template

parser = argparse.ArgumentParser(
    description = (
        "Evaluate a model, generating metrics and notebook reports. "
        "Use the optional '--compare_with' argument to compute additional comparisons between models."
    )
)

parser.add_argument(
    "-p", "--path", type=str, required=True, help="path to the folder of the run, ie, the folder that contains the hparams.json file"
)

parser.add_argument(
    "-c", "--compare_with", type=str, nargs="+", help="compare model with models from a list"
)

args = parser.parse_args()

assert os.path.isdir(args.path), f"{args.path} is not a folder"
EVAL_FOLDER = args.path.replace("models/", "eval/")

os.makedirs(EVAL_FOLDER, exist_ok=True)

log_info = get_log_info()

with open(f"{args.path}/hparams.json", "r") as f:
    model_hparams = json.load(f)


for pred_path in glob(f"{args.path}/predictions/*.jsonl"):

    n = 0
    error = 0
    subset_name = os.path.basename(pred_path).split(".")[0]

    sum_prediction = 0
    sum_target = 0
    sum_pred_times_target = 0

    with open(pred_path, "r") as f:
        for line in f.readlines():
            e = json.loads(line)
            n += 1
            sum_prediction += e["pred"] > .5
            sum_target += e["y"]
            sum_pred_times_target += (e["pred"] > .5) * e["y"]
            error += int(e["y"]) != int(e["pred"] > .5)

    out = {}
    if sum_prediction == 0:
        out["precision"] = 1
    else:
        out["precision"] = sum_pred_times_target / sum_prediction

    assert sum_target > 0, "{sum_target=} must be postive"
    out["recall"] = sum_pred_times_target / sum_target

    out["f-measure"] = 2.0 * sum_pred_times_target / (sum_target + sum_prediction)
    out["accuracy"] = 1 - error/n

    out["log"] = log_info
    out["log"]["model_log"] = model_hparams

    with open(f"{EVAL_FOLDER}/{subset_name}.json", "w") as f:
        json.dump(out, f, indent=4)


dict_args = vars(args)


dict_args = { k:str(dict_args[k]) for k in dict_args }
print(dict_args)
run_notebook_template("src/eval/notebook_templates/eval.ipynb" , dict_args, f"{EVAL_FOLDER}/eval.ipynb")
