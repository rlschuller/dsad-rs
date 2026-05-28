import argparse
import json
import os
import pathlib
import shutil
import sys

import torch
import torch.optim as optim
import torchinfo
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.utils.general import get_deterministic_name, get_hostname, get_git_revision, get_date
from src.utils.models import EarlyStopper, get_dataset, get_n_channels_time_series, get_n_channels

from src.utils.data import tensors_to_device
from src.utils.environ import DSAD_TQDM_NCOLS
from src.utils import board

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "-es",
    "--early_stop_patience",
    default=7,
    type=int,
)
parser.add_argument(
    "-nw",
    "--num_workers",
    default=8,
    type=int,
)
parser.add_argument(
    "-l",
    "--loss",
    default="bce",
    type=str,
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="adam",
    type=str,
)
parser.add_argument("-pw", "--pos_weight", default=None, type=float)
parser.add_argument("-lr", "--learning_rate", default=3e-4, type=float)
parser.add_argument("--momentum", default=0.0, type=float)
parser.add_argument("-wd", "--weight_decay", default=0.0, type=float)
parser.add_argument("-me", "--max_epochs", default=100, type=int)
parser.add_argument("-bs", "--batch_size", default=32, type=int)
parser.add_argument("-dbg_ini", "--debug_initialization", action='store_true')
parser.add_argument(
    "-aul",
    "--area_under_loss",
    action="store_true"
)
parser.add_argument(
    "--tag",
    type=str,
    help="tag to be saved in the hparams file"
)

parser_dataset = parser.add_argument_group("dataset")
parser_dataset.add_argument(
    "-d",
    "--dataset_path",
    default="data/processed/datasets/cloud_filter-closeup-time_series-205899.json",
    type=str,
)
parser_dataset.add_argument(
    "-b",
    "--bands",
    default=None, #["blue", "green", "red", "nir", "swir1", "swir2","red_edge_1","red_edge_2","red_edge_3", "red_edge_4", "clouds"],
    type=str,
    nargs="+",
    help="which bands to use. bands that are distinct in t0 and t1 will be added as two channels",
)
parser_dataset.add_argument(
    "--time_series_statistics",
    default=["mean", "stddev", "p25", "p50", "p75"],
    type=str,
    nargs="+",
    help="which statistics to select from the time series",
)
parser_dataset.add_argument(
    "-s",
    "--image_size",
    default=128,
    type=int,
    help="side of the square images. if image_size==None, use original resolution",
)
parser_dataset.add_argument(
    "-da", "--data_augmentation", action="store_true", help="augment dataset with flips and rotations"
)
parser_dataset.add_argument(
    "-ts",
    "--time_series",
    action="store_true",
)
parser_dataset.add_argument(
    "--time_series_folder_name",
    default="time_series",
    help="name of the default time_series folder, on 'data/processed'",
    type=str,
)

parser_dataset.add_argument(
    "--ignore_images",
    action="store_true",
    help="dont load images",
)
parser_dataset.add_argument('--no-drop_repeated_dates', dest='drop_repeated_dates', action='store_false')
parser.set_defaults(drop_repeated_dates=True)

parser_dataset.add_argument('--no-preload_labels', dest='preload_labels', action='store_false')
parser.set_defaults(preload_labels=True)

parser_dataset.add_argument('--no-preload_time_series', dest='preload_time_series', action='store_false')
parser.set_defaults(preload_time_series=True)

parser_dataset.add_argument(
    '-pd',
    '--pipes_dataset',
    action='store_true',
    help='use a fast, pre-processed dataset',
)

parser_dataset.add_argument(
    "--time_series_max_median_cloud_prob",
    default=None,
    type=int,
    help="remove entries in time series with clouds",
)
parser_dataset.add_argument(
    "--max_time_series_length",
    default=128,
    type=int,
    help="Truncate time series."
)

parser_dataset.add_argument(
    "--use_label_overrides",
    default=1,
    type=int,
)

parser_dataset.add_argument(
    "-ctx",
    "--context",
    action='store_true',
    help="include context images, ie, images with FOV larger than the alert itself",
)
parser_dataset.add_argument(
    "-ni", "--normalize_input", action='store_true', help="normalize input of the model"
)

parser_dataset.add_argument(
    "-i",
    "--input_override",
    default=None,
    nargs='*',
)

parser_dataset.add_argument(
    "-r",
    "--resample_prob_pos",
    default=None,
    type=float,
    help="resample training dataset, balancing the labels such that probabilty of beeing positive is resample_prob_pos",
)

parser_dataset.add_argument(
    "--mask_alert_geometry",
    action='store_true',
    help="make values outside the alert geometry 0",
)
parser_dataset.add_argument(
    "--mix_t0_and_t1",
    action='store_true',
    help="mix t0 and t1 based on parity (symmetry test)",
)

parser_system = parser.add_argument_group("system")
parser_system.add_argument(
    "--device",
    default="cuda",
    type=str,
)
parser_system.add_argument(
    "-pf", "--personal_folder", action="store_true", help="use personal folder instead of shared folder"
)

parser_system.add_argument(
    "-tb", "--tensorboard", action="store_true", help="Log training statistics with TensorBoard"
)
parser.set_defaults(tensorboard=False)

parser_system.add_argument(
    "-tbf", "--tensorboard_full", action="store_true", help="Log training FULL statistics with TensorBoard"
)
parser.set_defaults(tensorboard_full=False)

# Add model specific flags to parser
model_subparser = parser.add_subparsers(dest="model", metavar="MODEL", required=True)
model_names = sorted([path.parent.name for path in pathlib.Path.cwd().rglob("src/models/*/model.py")])
for model_name in model_names:
    if not model_name.startswith("_"):
        add_model_flags = getattr(
            __import__(f"src.models.{model_name}.model", fromlist=["add_subparser"]), "add_subparser"
        )
        add_model_flags(model_subparser)

args = parser.parse_args()

if args.tensorboard_full:
    args.tensorboard = True

# Create hparams dict with common parser args and model subparser args
all_args = vars(args)
common_args_keys = [action.dest for action in parser._actions if action.dest != "help"]
hparams = {k: all_args[k] for k in common_args_keys}
hparams["model_hparams"] = {k: all_args[k] for k in all_args if k not in common_args_keys}

bands = None
channels = None

if hparams["bands"] and (not hparams["input_override"] or len(hparams["input_override"]) == 1):
    # Compute bands and respective channels; add to hparams dict
    with open(hparams["dataset_path"], "r") as f:
        dataset_metadata = json.load(f)
        if "raw" in dataset_metadata["hparams"] and dataset_metadata["hparams"]["raw"]:
            available_bands_path = next(pathlib.Path(dataset_metadata["hparams"]["raw"]).rglob("*bands.json"))
        elif hparams["input_override"]:
            available_bands_path = next(pathlib.Path(f"data/processed/bands/{hparams['input_override'][0]}").glob("*.json"))
        else:
            available_bands_path = next(pathlib.Path(f"data/processed/bands/{dataset_metadata['hparams']['input']}").glob("*.json"))

    with available_bands_path.open("r") as f:
        available_bands = json.load(f)

    for b in hparams["bands"]:
        assert "_t0" not in b
        assert "_t1" not in b
        assert (b + "_t0") in available_bands or b in available_bands

    bands_in_t0_and_t1 = [b for b in hparams["bands"] if b + "_t0" in available_bands]
    bands_in_t0_and_t1 = [b + "_t0" for b in bands_in_t0_and_t1] + [b + "_t1" for b in bands_in_t0_and_t1]
    single_bands = [b for b in hparams["bands"] if b in available_bands]

    bands = [b for b in available_bands if b in bands_in_t0_and_t1 or b in single_bands]
    channels = [i for i, b in enumerate(available_bands) if b in bands_in_t0_and_t1 or b in single_bands]

hparams["computed_hparams"] = {}
hparams["computed_hparams"]["bands"] = bands
hparams["computed_hparams"]["channels"] = channels

# Create output folders and save hparams
date_start = get_date()
run_name = date_start + "-" + get_deterministic_name(hparams)
base_folder = "personal" if hparams["personal_folder"] else "."
output_folder = f"{base_folder}/models/{hparams['model']}/{run_name}"

# tensorboard logger will honor 'personal folder' flag
tb_log_folder = f"{output_folder}/tensorboard"


if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

print(f"{output_folder=}")
checkpoints_folder = f"{output_folder}/checkpoints"
loss_folder = f"{output_folder}/loss"
os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(loss_folder, exist_ok=True)

info = {}
with open(output_folder + "/info.json", "w") as f:
    info["hostname"] = get_hostname()
    info["git"] = get_git_revision()
    info["date_start"] = date_start
    info["argv"] = sys.argv
    f.write(json.dumps(info, indent=4)+"\n")

# by default, copytree attempts to preserve metadata, since it is
# a recursive call of the copy2 function.
shutil.copytree("data/processed/label_reviews", f"{output_folder}/label_reviews")
hparams["computed_hparams"]["label_reviews_folder"] = f"{output_folder}/label_reviews"

# Load data and create dataloaders
g = torch.Generator().manual_seed(0)

train_dataset = get_dataset(hparams, subset="train")
val_dataset = get_dataset(hparams, subset="val")

train_dataloader = DataLoader(
    train_dataset,
    hparams["batch_size"],
    shuffle=True,
    num_workers=hparams["num_workers"],
    generator=g,
    # pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    hparams["batch_size"],
    shuffle=False,
    num_workers=hparams["num_workers"],
    generator=g,
    # pin_memory=True,
)

# Load Model from the corresponding file
Model = getattr(__import__(f"src.models.{hparams['model']}.model", fromlist=["Model"]), "Model")
device = hparams["device"]
print(f"{device=}")

hparams["computed_hparams"]["n_channels"] = get_n_channels(train_dataset)
hparams["computed_hparams"]["n_channels_time_series"] = get_n_channels_time_series(train_dataset)

with open(output_folder + "/hparams.json", "w") as f:
    hparams["output_folder"] = output_folder
    f.write(json.dumps(hparams, indent=4)+"\n")

if hparams["time_series"]:
    model = Model(
        hparams=hparams,
        n_channels=hparams["computed_hparams"]["n_channels"],
        n_channels_time_series=hparams["computed_hparams"]["n_channels_time_series"],
        length_time_series=hparams["max_time_series_length"],
    )

else:
    model = Model(
        hparams=hparams,
        n_channels=hparams["computed_hparams"]["n_channels"],
    )
model.to(device)

early_stop = EarlyStopper(patience=hparams["early_stop_patience"])

if hparams["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
elif hparams["optimizer"] == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"])
else:
    raise NotImplementedError

if hparams["pos_weight"]:
    pos_weight = torch.tensor([float(hparams["pos_weight"])], dtype=torch.float32).to(device)
else:
    pos_weight = None

if hparams["loss"] == "bce":
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
else:
    raise NotImplementedError


if hparams["time_series"]:
    # torchinfo_data = (train_dataset[0][0][0].unsqueeze(0),train_dataset[0][0][1].unsqueeze(0))
    torchinfo.summary(model)
else:
    torchinfo.summary(model)#, input_size=train_dataset[0][0].unsqueeze(0).shape)


# create tensorboard logger
if hparams["tensorboard"]:
    tb_writer = SummaryWriter(tb_log_folder)
    if hparams["tensorboard_full"]:
        data, label, _, _ = next(iter(train_dataloader))
        if hparams["time_series"]:
            data = (data[0].to(device), data[1].to(device)),  # yes, a tuple, go figure...
        else:
            data = data.to(device)
        tb_writer.add_graph(model, data)

if hparams["area_under_loss"]:
    aul_folder = f"{output_folder}/predictions-area_under_loss"
    os.makedirs(aul_folder, exist_ok=True)

metrics = dict({
    "best_train_loss": float("inf"),
    "best_validation_loss": float("inf"),
    "best_train_accuracy": float(0.),
    "best_validation_accuracy": float(0.),
})

try:
    torch.manual_seed(0)
    for epoch in range(hparams["max_epochs"]):
        if hparams["resample_prob_pos"]:
            train_dataset.resample(prob_pos=hparams["resample_prob_pos"])
        print(f"{epoch=}")
        acc_loss = 0
        correct_preds = 0

        for X, y, path, i in tqdm(train_dataloader, ncols=DSAD_TQDM_NCOLS):#, ncols=100):
            X = tensors_to_device(X, device)
            y = tensors_to_device(y, device)

            # Use epoch 0 to test the initialization
            if epoch > 0 or not hparams["debug_initialization"]:
                optimizer.zero_grad()
                model.train()

                #with torch.autocast(device_type=hparams["device"]):
                pred = model(X)
                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()

            else:
                with torch.no_grad():
                    # make sure batchnorm is off
                    model.eval()
                    #with torch.autocast(device_type=hparams["device"]):
                    pred = model(X)
                    loss = loss_func(pred, y)

            acc_loss += loss.item()

            with torch.no_grad():
                if hparams["area_under_loss"]:

                    for j in range(i.shape[0]):
                        entry = {
                            "epoch":epoch,
                            "i":i[j].item(),
                            "path":path[j],
                            "pred":pred[j][0].tolist(),
                            "y":y[j][0].tolist(),
                            "loss": loss_func(pred[j], y[j]).item()
                        }
                        with open(f"{aul_folder}/train.jsonl", "a") as f:
                            f.write(json.dumps(entry)+"\n")

                if hparams["loss"] == "bce":
                    pred = torch.sigmoid(pred)
                correct_preds += torch.sum((pred > 0.5) == y).item()


        train_loss = acc_loss / len(train_dataloader)
        metrics["best_train_loss"] = min(metrics["best_train_loss"], train_loss)
        train_accuracy = float(correct_preds) / len(train_dataset)
        metrics["best_train_accuracy"] = max(metrics["best_train_accuracy"], train_accuracy)

        print(f"{train_loss=}")
        with open(f"{loss_folder}/train-loss.jsonl", "a") as f:
            f.write(str(train_loss) + "\n")

        model.eval()
        acc_loss = 0
        correct_preds = 0
        predictions = {
            "all": list(),
            "0": list(),
            "1": list(),
        }
        indices = {
            "0": list(),
            "1": list(),
        }
        with torch.inference_mode():
            for X, y, _, I in tqdm(val_dataloader, ncols=DSAD_TQDM_NCOLS):#, ncols=100):
                X = tensors_to_device(X, device)
                y = tensors_to_device(y, device)

                I = I.to(device)
                #with torch.autocast(device_type=hparams["device"]):
                pred = model(X)
                loss = loss_func(pred, y)
                if hparams["loss"] == "bce":
                    pred = torch.sigmoid(pred)
                correct_preds += torch.sum((pred > 0.5) == y).item()
                acc_loss += loss.item()
                # save batch predictions
                if hparams["tensorboard_full"]:
                    predictions["all"].append(pred)
                    # 0
                    idx = torch.nonzero(y == 0, as_tuple=False)[:, 0]
                    predictions["0"].append(pred[idx])
                    indices["0"].append(I[idx])
                    # 1
                    idx = torch.nonzero(y == 1, as_tuple=False)[:, 0]
                    predictions["1"].append(pred[idx])
                    indices["1"].append(I[idx])

        val_loss = acc_loss / len(val_dataloader)
        metrics["best_validation_loss"] = min(metrics["best_validation_loss"], val_loss)
        val_accuracy = float(correct_preds) / len(val_dataset)
        metrics["best_validation_accuracy"] = max(metrics["best_validation_accuracy"], val_accuracy)

        if hparams["tensorboard"]:
            tb_writer.add_scalars("Curves/Loss", {
                    "Train": train_loss,
                    "Validation": val_loss,
                },
                epoch
            )
            tb_writer.add_scalars("Curves/Accuracy", {
                    "Train": train_accuracy,
                    "Validation": val_accuracy,
                },
                epoch
            )
            if hparams["tensorboard_full"]:
                for name, params in model.named_parameters():
                    # print(name, params)
                    tb_writer.add_histogram(f"Layer Gradients/{name}", params.grad, epoch)
                # save histogram of predictions
                num_bins = 20 
                tb_writer.add_histogram(
                    f"Predictions Distribution/All",
                    torch.vstack(predictions["all"]).flatten(),
                    epoch,
                    bins=num_bins,
                )
                # 0
                pred_0 = torch.vstack(predictions["0"]).flatten()
                ids_0 = torch.hstack(indices["0"]).flatten()
                tb_writer.add_histogram(
                    f"Predictions Distribution/Label 0",
                    pred_0,
                    epoch,
                    bins=num_bins,
                )
                figs = board.get_histogram_with_images(pred_0, ids_0, val_dataset, num_bins=num_bins, device=device)
                for i, (fig, val) in enumerate(figs):
                    tb_writer.add_figure(f"Images/E[{epoch}]/Label[0]", fig, global_step=i+1)
                # 1
                pred_1 = torch.vstack(predictions["1"]).flatten()
                ids_1 = torch.hstack(indices["1"]).flatten()
                tb_writer.add_histogram(
                    f"Predictions Distribution/Label 1",
                    pred_1,
                    epoch,
                    bins=num_bins,
                )
                figs = board.get_histogram_with_images(pred_1, ids_1, val_dataset, num_bins=num_bins, device=device)
                for i, (fig, val) in enumerate(figs):
                    tb_writer.add_figure(f"Images/E[{epoch}]/Label[1]", fig, global_step=i+1)

        print(f"{val_loss=}")
        print(f"{val_accuracy=}")
        print()

        with open(f"{loss_folder}/val-loss.jsonl", "a") as f:
            f.write(str(val_loss) + "\n")
        with open(f"{loss_folder}/val-accuracy.jsonl", "a") as f:
            f.write(str(val_accuracy) + "\n")

        torch.save(
            model.state_dict(),
            f"{checkpoints_folder}/epoch_{epoch}-val_loss_{val_loss}.ckp",
        )

        if early_stop(val_loss):
            print("Stopping early")
            break

    with open(output_folder + "/info.json", "w") as f:
        info["date_end"] = get_date()
        f.write(json.dumps(info, indent=4)+"\n")
except KeyboardInterrupt:
    print("Keyboard Interruption. Exiting....")

finally:
    print("Saving hparams!")
    tb_hparams = {}
    for k, v in hparams.items():
        # print(k, type(v))
        if isinstance(v, list):
            v = str(v)
        elif isinstance(v, dict):
            v = json.dumps(v, indent=4)
        tb_hparams[k] = v

    if hparams["tensorboard"]:
        tb_writer.add_hparams(
            hparam_dict=tb_hparams,
            metric_dict=metrics,
            run_name=run_name
        )
    print("HParams saved!")
    print(f"{output_folder=}")

