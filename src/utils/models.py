import numpy as np
import json
from glob import glob
import torch
from src.utils.data import HdfDataset, PipesDataset, CombineDatasets, RotateAndFlip
from typing import List, Tuple, Dict
from pathlib import Path

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class AddSigmoid(torch.nn.Module):
    def __init__(self, model):
        super(AddSigmoid, self).__init__()
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()

        if "cam_target_layers" in model._modules:
            self.cam_target_layers = self.model.cam_target_layers

    def forward(self, x):
        if self.training:
            return self.model(x)
        else:
            return self.sigmoid(self.model(x))

def get_best_model(train_folder, device='cuda', use_accuracy=True, disable_sigmoid=False, checkpoint_path=None):

    with open(f"{train_folder}/hparams.json", "r") as f:
        hparams = json.load(f)

    checkpoints_folder = f"{train_folder}/checkpoints"
    Model = getattr(__import__(f"src.models.{hparams['model']}.model", fromlist=["Model"]), "Model")


    if "n_channels" not in hparams["computed_hparams"]:
        d = get_dataset(hparams)
        hparams["computed_hparams"]["n_channels"] = get_n_channels(d)

    if "n_channels_time_series" not in hparams["computed_hparams"]:
        d = get_dataset(hparams)
        hparams["computed_hparams"]["n_channels_time_series"] = get_n_channels_time_series(d)

    if "time_series" in hparams and hparams["time_series"]:

        model = Model(
            hparams=hparams,
            n_channels=hparams["computed_hparams"]["n_channels"],
            n_channels_time_series=hparams["computed_hparams"]["n_channels_time_series"],
            length_time_series=hparams["max_time_series_length"]
        )
    else:
        model = Model(
            hparams=hparams,
            n_channels=hparams["computed_hparams"]["n_channels"],
        )


    acc_history = [float(x) for x in Path(f"{train_folder}/loss/val-accuracy.jsonl").open("r").readlines()]
    best_epoch = np.argmax(acc_history)
    min_acc_path = glob(f"{checkpoints_folder}/epoch_{best_epoch}*.ckp")[0]
    min_loss = 9999999
    min_loss_path = ""
    for _checkpoint_path in glob(f"{checkpoints_folder}/*.ckp"):
        loss = float(_checkpoint_path.split("_")[-1].replace(".ckp", ""))
        if loss < min_loss:
            min_loss = loss
            min_loss_path = _checkpoint_path

    state_keys = torch.load(min_acc_path).keys()

    if checkpoint_path:
        min_acc_path = checkpoint_path
        min_loss_path = checkpoint_path

    print(f"{min_acc_path=}")

    if "cam_target_layers" in  list(vars(model)['_modules'].keys()):
        cam_target_layers = model.cam_target_layers

        if not any(['cam_target_layers' in k for k in state_keys]):
            print("saved checkpoint has no cam_target_layers")
            model.cam_target_layers = None

        if use_accuracy:
            print(f"{acc_history[best_epoch]=}")
            model.load_state_dict(torch.load(min_acc_path, map_location=torch.device(device)))
        else:
            print(f"{min_loss=}")
            model.load_state_dict(torch.load(min_loss_path, map_location=torch.device(device)))


        model.cam_target_layers = cam_target_layers
    else:

        if use_accuracy:
            print(f"{acc_history[best_epoch]=}")
            model.load_state_dict(torch.load(min_acc_path, map_location=torch.device(device)))
        else:
            print(f"{min_loss=}")
            model.load_state_dict(torch.load(min_loss_path, map_location=torch.device(device)))

    # if using BCE, add sigmoid to model
    if hparams["loss"] == "bce" and not disable_sigmoid:
        model = AddSigmoid(model)

    model.to(device)
    model.eval()

    return model


def get_n_channels(dataset):

    # time series
    if isinstance(dataset[0][0], Tuple) or isinstance(dataset[0][0], List):
        X_0 = dataset[0][0][1]
    else:
        X_0 = dataset[0][0]

    if isinstance(X_0, Dict):
        print("a dictionary!")

        n_channels = {}

        for k in X_0:
            n_channels[k] = X_0[k].unsqueeze(0).shape[1]
        print(json.dumps(n_channels, indent=4))
        return n_channels

    else:
        return X_0.unsqueeze(0).shape[1]

def get_n_channels_time_series(dataset):
    if isinstance(dataset[0][0], Tuple) or isinstance(dataset[0][0], List):
        t_0 = dataset[0][0][0]
        return t_0.unsqueeze(0).shape[1]
    else:
        return None


def get_dataset(hparams, subset='test'):
    channels = hparams["computed_hparams"]["channels"]

    # Default normalization to False, for bacwards compatibility
    if "normalize_input" not in hparams:
        hparams["normalize_input"] = False

    if "drop_repeated_dates" not in hparams:
        hparams["drop_repeated_dates"] = False

    if "time_series" not in hparams:
        hparams["time_series"] = False

    if "time_series_statistics" not in hparams:
        hparams["time_series_statistics"] = None

    if 'time_series_max_median_cloud_prob' not in hparams:
        hparams['time_series_max_median_cloud_prob'] = None

    if 'ignore_images' not in hparams:
        hparams['ignore_images'] = None

    if 'preload_time_series' not in hparams:
        hparams['preload_time_series'] = False

    if 'preload_labels' not in hparams:
        hparams['preload_labels'] = False

    if "max_time_series_length" not in hparams:
        hparams["max_time_series_length"] = 128

    if "use_label_overrides" not in hparams:
        hparams["use_label_overrides"] = True

    if "time_series_folder_name" not in hparams:
        hparams["time_series_folder_name"] = "time_series"

    if "label_reviews_folder" not in hparams["computed_hparams"]:
        hparams["computed_hparams"]["label_reviews_folder"] = "data/processed/label_reviews/"

    if "data_augmentation" not in hparams:
        hparams["data_augmentation"] = False

    if "input_override" not in hparams:
        hparams["input_override"] = None

    if "mask_alert_geometry" not in hparams:
        hparams["mask_alert_geometry"] = False

    if "mix_t0_and_t1" not in hparams:
        hparams["mix_t0_and_t1"] = False

    assert ("context" not in hparams) or (not hparams["context"])
    assert ("pipes_dataset" not in hparams) or (not hparams["pipes_dataset"])

    if hparams["input_override"] and isinstance(hparams["input_override"], List) and len(hparams["input_override"]) > 1:
        dataset_list = []
        for input in hparams["input_override"]:
            dataset = HdfDataset(
                dataset_path=hparams["dataset_path"],
                subset=subset,
                image_size=hparams["image_size"],
                normalize_input=hparams["normalize_input"],
                input_override=input,
                use_time_series=False,
                max_time_series_length=hparams["max_time_series_length"],
                use_label_overrides=hparams["use_label_overrides"],
                time_series_statistics=hparams["time_series_statistics"],
                max_median_cloud_prob = hparams["time_series_max_median_cloud_prob"],
                ignore_images=hparams["ignore_images"],
                drop_repeated_dates=hparams["drop_repeated_dates"],
                time_series_folder_name=hparams["time_series_folder_name"],
                preload_time_series=hparams["preload_time_series"],
                preload_labels=hparams["preload_labels"],
                mask_alert_geometry=hparams["mask_alert_geometry"],
                label_reviews_folder=hparams["computed_hparams"]["label_reviews_folder"],
                mix_t0_and_t1=hparams["mix_t0_and_t1"],
            )
            dataset_list.append(dataset)

        time_series_dataset=None
        if hparams["time_series"]:
            time_series_dataset = HdfDataset(
                dataset_path=hparams["dataset_path"],
                subset=subset,
                image_size=hparams["image_size"],
                normalize_input=hparams["normalize_input"],
                input_override=None,
                use_time_series=True,
                max_time_series_length=hparams["max_time_series_length"],
                use_label_overrides=hparams["use_label_overrides"],
                time_series_statistics=hparams["time_series_statistics"],
                max_median_cloud_prob = hparams["time_series_max_median_cloud_prob"],
                ignore_images=True,
                drop_repeated_dates=hparams["drop_repeated_dates"],
                time_series_folder_name=hparams["time_series_folder_name"],
                preload_time_series=hparams["preload_time_series"],
                preload_labels=hparams["preload_labels"],
                label_reviews_folder=hparams["computed_hparams"]["label_reviews_folder"],
                mix_t0_and_t1=hparams["mix_t0_and_t1"],
            )

        default_dataset = CombineDatasets(
            dataset_list=dataset_list,
            time_series_dataset=time_series_dataset,
        )

    else:
        if hparams["input_override"] and isinstance(hparams["input_override"], List):
            hparams["input_override"] = hparams["input_override"][0]

        default_dataset = HdfDataset(
            dataset_path=hparams["dataset_path"],
            subset=subset,
            image_size=hparams["image_size"],
            channels=channels,
            normalize_input=hparams["normalize_input"],
            input_override=hparams["input_override"],
            use_time_series=hparams["time_series"],
            max_time_series_length=hparams["max_time_series_length"],
            use_label_overrides=hparams["use_label_overrides"],
            time_series_statistics=hparams["time_series_statistics"],
            max_median_cloud_prob = hparams["time_series_max_median_cloud_prob"],
            ignore_images=hparams["ignore_images"],
            drop_repeated_dates=hparams["drop_repeated_dates"],
            time_series_folder_name=hparams["time_series_folder_name"],
            preload_time_series=hparams["preload_time_series"],
            preload_labels=hparams["preload_labels"],
            mask_alert_geometry=hparams["mask_alert_geometry"],
            label_reviews_folder=hparams["computed_hparams"]["label_reviews_folder"],
            mix_t0_and_t1=hparams["mix_t0_and_t1"],
        )

    dataset = default_dataset

    if hparams["data_augmentation"]:
        if hparams["input_override"] and isinstance(hparams["input_override"], List):
            raise NotImplementedError
        return RotateAndFlip(dataset)

    return dataset

