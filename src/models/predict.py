import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from src.utils.general import get_deterministic_name
from src.utils.data import HdfDataset, RawTiffDataset
from src.utils.data import tensors_to_device
from src.utils.models import get_best_model, get_dataset
from src.utils.environ import DSAD_TQDM_NCOLS

parser = argparse.ArgumentParser(
    description = "Save classification predictions as jsonl files."
)

parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="path of the run's folder",
)
parser.add_argument(
    "-nw",
    "--num_workers",
    default=16,
    type=int,
)

parser.add_argument(
    "-t",
    "--tifs_folder",
    type=str,
    help="read inputs and targets from tifs folder instead of original dataset",
)

parser.add_argument(
    "-ds",
    "--disable_sigmoid",
    action="store_true",
    help="disable last sigmoid layer and save predictions in 'predictions-without_sigmoid'",
)

parser.add_argument(
    "-o",
    "--output_folder",
    type=str,
    help="override output folder"
)

parser.add_argument(
    "-aul",
    "--area_under_loss",
    action="store_true",
    help="compute area under loss for training subset (compute intensive)",
)

parser.add_argument(
    "--dry",
    action="store_true",
    help="only print summary instead of running predictions",
)


parser.add_argument("--device", default="cuda", type=str)

args = parser.parse_args()


if args.output_folder:
    assert args.tifs_folder

device = args.device

with open(f"{args.path}/hparams.json", "r") as f:
    hparams = json.load(f)

base_folder = "personal" if hparams["personal_folder"] else "."
predictions_folder = f"{args.path}/predictions"
if args.disable_sigmoid:
    predictions_folder = predictions_folder + "-without_sigmoid"

checkpoints_folder = f"{args.path}/checkpoints"
print(f"{predictions_folder=}")
os.makedirs(predictions_folder, exist_ok=True)

print(json.dumps(hparams, indent=4))

model = get_best_model(args.path, disable_sigmoid=args.disable_sigmoid)

if args.dry:
    hparams.pop('preload_time_series')
    hparams.pop('preload_labels')
    dataset = get_dataset(hparams, subset="train")
    dataloader = DataLoader(dataset, 1)#args.num_workers)

    model.to(args.device)
    for X, y, path, _ in tqdm(dataloader, ncols=DSAD_TQDM_NCOLS):

        X = tensors_to_device(X, device)
        print(X)
        summary(model, input_data=X)
        exit(0)

with torch.no_grad():
    if args.tifs_folder:
        if args.disable_sigmoid:
            raise NotImplementedError("currently, there is no convention establishing where to save predictions without sigmoid when the inputs are tif files.")

        if args.output_folder:
            OUTPUT_FOLDER = Path(args.output_folder)
        else:
            OUTPUT_FOLDER = Path("data/uploads") / Path(args.tifs_folder).name / hparams["model"] / Path(hparams["output_folder"]).name

        (OUTPUT_FOLDER / "predictions").mkdir(exist_ok=True, parents=True)
        shutil.copy(f"{args.path}/hparams.json", OUTPUT_FOLDER)

        print(f"{OUTPUT_FOLDER}")
        f = open(f"{OUTPUT_FOLDER}/predictions/all.jsonl", "w")

        if "normalize_input" in hparams:
            normalize_input = True
        else:
            normalize_input = False

        dataset = RawTiffDataset(
            args.tifs_folder, image_size=hparams["image_size"], bands=hparams["computed_hparams"]["bands"], normalize_input=normalize_input
        )

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=args.num_workers)

        correct_preds = 0
        total = len(dataset)

        for X, y, path, _ in tqdm(dataloader, ncols=DSAD_TQDM_NCOLS):

            X = tensors_to_device(X, device)
            y = tensors_to_device(y, device)

            #with torch.autocast(device_type=hparams["device"]):
            pred = model(X)

            correct_preds += torch.sum((pred > 0.5) == y).item()
            y = y.to("cpu")
            pred = pred.to("cpu")

            for i in range(y.shape[0]):
                _y = float(y[i][0])
                _pred = float(pred[i][0])
                _path = path[i]
                f.write(json.dumps({"y": _y, "pred": _pred, "path": path[i]}) + "\n")


    else:
        hparams["data_augmentation"] = False

        if args.area_under_loss:
            aul_folder = f"{predictions_folder}-area_under_loss"

            os.makedirs(aul_folder, exist_ok=True)
            f = open(f"{aul_folder}/train.jsonl", "w")
            dataset = get_dataset(hparams, subset="train")
            dataloader = DataLoader(dataset, 32, shuffle=False, num_workers=8)#args.num_workers)

            checkpoint_paths = sorted(list(glob(f"{checkpoints_folder}/*.ckp")), key = lambda x : int(x.split("/")[-1].split("-")[0].split("_")[1]))
            loss = nn.BCELoss()

            for checkpoint_path in checkpoint_paths:
                epoch = int(checkpoint_path.split("/")[-1].split("-")[0].split("_")[1])
                model_aul = get_best_model(args.path, disable_sigmoid=args.disable_sigmoid, checkpoint_path = checkpoint_path)
                for X, y, path, _ in tqdm(dataloader, desc=f"e={epoch}", ncols=DSAD_TQDM_NCOLS):
                    X = tensors_to_device(X, device)
                    y = tensors_to_device(y, device)

                    #with torch.autocast(device_type=hparams["device"]):
                    pred = model_aul(X)

                    y = y.to("cpu")
                    pred = pred.to("cpu")

                    for i in range(y.shape[0]):
                        _y = float(y[i][0])
                        _pred = float(pred[i][0])
                        _loss = float(loss(pred[i][0], y[i][0]))
                        f.write(json.dumps({"y": _y, "pred": _pred, "path": path[i], "checkpoint":checkpoint_path, "epoch":epoch, "loss":_loss}) + "\n")
            f.close()


        for subset in ["test", "val", "train"]:

            f = open(f"{predictions_folder}/{subset}.jsonl", "w")
            dataset = get_dataset(hparams, subset=subset)
            dataloader = DataLoader(dataset, 32, shuffle=False, num_workers=8)#args.num_workers)

            correct_preds = 0
            total = len(dataset)
            for X, y, path, _ in tqdm(dataloader, desc=subset, ncols=DSAD_TQDM_NCOLS):
                X = tensors_to_device(X, device)
                y = tensors_to_device(y, device)

                #with torch.autocast(device_type=hparams["device"]):
                pred = model(X)

                correct_preds += torch.sum((pred > 0.5) == y).item()
                y = y.to("cpu")
                pred = pred.to("cpu")

                for i in range(y.shape[0]):
                    _y = float(y[i][0])
                    _pred = float(pred[i][0])
                    _path = path[i]
                    f.write(json.dumps({"y": _y, "pred": _pred, "path": path[i]}) + "\n")

            if not args.disable_sigmoid:
                accuracy = correct_preds / total
                print(f"{accuracy=}")

            f.close()

