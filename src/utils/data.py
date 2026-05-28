import json
import os
import io
from pathlib import Path
import inspect

import subprocess
import time
from typing import List, Tuple, Dict

import numpy as np
import tifffile
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import h5py
import PIL
from tqdm import tqdm
from src.utils.environ import DSAD_TQDM_NCOLS

import datetime

class RawTiffDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tifs_folder="data/raw/202308/",
        data_type=torch.float32,
        image_size=None,
        bands=None,
        normalize_input=True,
    ):

        available_bands = json.load(next(Path(tifs_folder).glob("*bands.json")).open("r"))
        self.bands = bands
        assert set(bands).issubset(set(available_bands))

        self.channels = [i for i, b in enumerate(available_bands) if b in bands]
        self.paths = sorted([str(p) for p in Path(tifs_folder).glob("*.tif")])


        self.data_type = data_type
        self.image_size = image_size

        if self.image_size:
            self.transform = T.Resize((image_size, image_size))

        self.normalize_input = normalize_input

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image_data = np.array(tifffile.imread(self.paths[i]), dtype=np.float32)

        image = torch.tensor(image_data, dtype=self.data_type)
        # put channels first
        image = image.permute(2, 0, 1)

        if self.image_size:
            image = self.transform(image)

        if self.channels:
            image = image[self.channels, :, :]

        if self.normalize_input:
            image = image / max(torch.max(image), 1)

        label_path = Path(self.paths[i].replace(".tif", "-label.json"))

        label = json.load(label_path.open("r"))
        if not label:
            label = -1
        label = torch.tensor([label], dtype=self.data_type)
        return image, label, self.paths[i]

class TiffDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path="data/processed/datasets/8089-harlequin_lamprey.json",
        subset="test",
        data_type=torch.float32,
        image_size=None,
        channels=None,
        normalize_input=True,
    ):

        # glob tif and tiff files
        with open(dataset_path, "r") as f:
            self.paths = json.load(f)[subset]

        self.data_type = data_type
        self.image_size = image_size

        if self.image_size:
            self.transform = T.Resize((image_size, image_size))

        self.channels = channels

        self.normalize_input = normalize_input
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image_data = np.array(tifffile.imread(self.paths[i]), dtype=np.float32)

        image = torch.tensor(image_data, dtype=self.data_type)
        # put channels first
        image = image.permute(2, 0, 1)

        if self.image_size:
            image = self.transform(image)

        if self.channels:
            image = image[self.channels, :, :]

        if self.normalize_input:
            image = image / max(torch.max(image), 1)

        label_path = Path(self.paths[i].replace(".tif", "-label.json"))
        label = json.load(label_path.open("r"))

        label = torch.tensor([label], dtype=self.data_type)
        return image, label, self.paths[i]

def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

class PipesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path="data/processed/datasets/cloud_filter-closeup-114030.json",
        subset="test",
        data_type=torch.float16,
        image_size=None,
        channels=None,
        normalize_input=True,
        input_override=None, #NOT SUPPORTED
        use_time_series=False,
        max_time_series_length=128,
        use_label_overrides=True, # use imazon's review
        time_series_statistics=["mean", "stddev", "p25", "p50", "p75"], #NOT SUPPORTED
        preload_time_series=False, #DOES NOT APPLY
        max_median_cloud_prob=None, # 0 to 100 or None #NOT SUPPORTED
        ignore_images=False,
        preload_labels=False, #DOES NOT APPLY
        drop_repeated_dates=False, #NOT SUPPORTED
        time_series_folder_name="time_series", #NOT SUPPORTED
        label_reviews_folder="data/processed/label_reviews/",
    ):
        format_version = "1.0"
        self.format_version = format_version
        self.folder = Path(f"data/processed/pipes_dataset/") / Path(dataset_path).stem
        self.subset = subset
        kwargs = get_kwargs()

        # verify hparams consistency

        with Path(f"{self.folder}/{self.subset}-hparams.json").open("r") as f:
            original_kwargs = json.load(f)
            original_kwargs["init_signature"].update(original_kwargs["kwargs"])
            original_kwargs = original_kwargs["init_signature"]

        kwargs_to_verify = ["input_override", "time_series_statistics", "max_median_cloud_prob", "drop_repeated_dates", "time_series_folder_name"]

        for k in kwargs:
            assert k not in kwargs_to_verify or str(kwargs[k]) == str(original_kwargs[k]), f"hparam {k}={kwargs[k]} is different than original hparam {original_kwargs[k]}"

        # glob tif and tiff files
        with open(dataset_path, "r") as f:
            dataset_metadata = json.load(f)
            self.paths = dataset_metadata[subset]

        self.folder = Path(f"data/processed/pipes_dataset/") / Path(dataset_path).stem
        self.subset = subset

        info = json.load(Path(f"{self.folder}/info.json").open("r"))
        assert format_version == info["format_version"], (
            f"PipesDataset tried to use format version {format_version}, "
            f"while data is stored in format version {info['format_version']}. "
             "Cowardly refusing..."
        )

        s = time.perf_counter()
        self.offsets = [int(x) for x in Path(f"{self.folder}/{self.subset}.offsets").open("r").readlines()]
        self.labels = [int(x) for x in Path(f"{self.folder}/{self.subset}.labels").open("r").readlines()]
        self._preloaded_data = torch.load(f"{self.folder}/{self.subset}.time_series")
        e = time.perf_counter()
        print(f"preloaded labels & time series in {(e-s)*1000:.2f} ms")

        self.max_time_series_length = max_time_series_length

        assert max_time_series_length <= self._preloaded_data.shape[2], f"pre-cached time series (length={self._preloaded_data[2]}) is not long enough"

        self.img_path = f"{self.folder}/{self.subset}.img"


        hdf_paths = Path(f"data/processed/hdf/{dataset_metadata['hparams']['input']}/").glob("*.hdf")
        with open(str(list(hdf_paths)[0]).replace("/hdf/", "/bands/").replace(".hdf",".json"), "r") as f:
            self.bands = json.load(f)
        if channels:
            self.bands = [self.bands[c] for c in channels]

        self.data_type = data_type
        self.image_size = image_size

        if self.image_size:
            self.transform = T.Resize((image_size, image_size))

        self.normalize_input = normalize_input
        self.use_time_series = use_time_series

        self.channels = channels
        self.label_overrides = {}

        if use_label_overrides:
            for path in sorted(Path(label_reviews_folder).glob("*.json")):
                with path.open("r") as f:
                    label_overrides = json.load(f)
                    self.label_overrides.update(label_overrides)
            print(f"label overrides loaded from {label_reviews_folder}")
        self.ignore_images = ignore_images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        path = self.paths[i]
        image_id = path.split("/")[-1].split(".")[0]

        if self.use_time_series:
            time_series = self._preloaded_data[i]
            if self.max_time_series_length:
                time_series = time_series[:, -self.max_time_series_length:].type(self.data_type)


        proc = subprocess.Popen(f"tail -c +{self.offsets[i]+1} {self.img_path} | head -c {self.offsets[i+1]-self.offsets[i]}", stdout=subprocess.PIPE, shell=True)
        buffer = proc.communicate()[0]

        shape  = np.frombuffer(buffer, dtype=np.int16, count=3)
        image  = np.frombuffer(buffer, dtype=np.int16, count=int(shape[0])*int(shape[1])*int(shape[2]), offset=3*2)
        image = torch.tensor(image, dtype=torch.float32).view(shape[0], shape[1], shape[2])

        if self.normalize_input:
            image = image / max(torch.max(image), 1)

        if self.channels:
            image = image[self.channels, :, :]

        if self.image_size:
            with torch.no_grad():
                #image = self.transform(image)
                image = F.interpolate(input=image.unsqueeze(0), size=self.image_size)
                image = image.squeeze()

        #image =image.float()

        label = self.labels[i]

        if image_id in self.label_overrides:
            label = self.label_overrides[image_id]

        label = torch.tensor([label], dtype=self.data_type)
        if self.data_type != torch.float16:
            image = image.type(self.data_type)

        if self.use_time_series:
            # if self.max_time_series_length:
                # time_series = time_series[:, -self.max_time_series_length:]

            return (time_series, image), label, path

        else:
            return image, label, path

def hparams_to_self(init):
    def store_hparams_wrapper(*args, **kwargs):
        assert len(args) == 0, "@store_hparams decorator does not support args, only kwargs"
        d = init(*args, **kwargs)
        signature = inspect.signature(init)
        d.hparams = {"init_signature": {
                k: str(v.default)
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            },
            "kwargs":{str(k):str(kwargs[k]) for k in kwargs}
         }
        return d


    return store_hparams_wrapper

@hparams_to_self
class HdfDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path="data/processed/datasets/cloud_filter-closeup-114030.json",
        subset="test",
        data_type=torch.float32,
        image_size=None,
        channels=None,
        normalize_input=True,
        input_override=None,
        use_time_series=False,
        max_time_series_length=128,
        mask_alert_geometry=False,
        use_label_overrides=True, # use imazon's review
        time_series_statistics=["mean", "stddev", "p25", "p50", "p75"],
        preload_time_series=False,
        max_median_cloud_prob=None, # 0 to 100 or None
        ignore_images=False,
        preload_labels=False,
        drop_repeated_dates=True,
        time_series_folder_name="time_series",
        label_reviews_folder="data/processed/label_reviews/",
        mix_t0_and_t1=False,
    ):


        # glob tif and tiff files
        with open(dataset_path, "r") as f:
            dataset_metadata = json.load(f)
            self.paths = dataset_metadata[subset]

        if input_override:
            dataset_metadata['hparams']['input'] = input_override

        if dataset_metadata['hparams']['input'] != "closeup" and mix_t0_and_t1:
            raise NotImplementedError

        self.mix_t0_and_t1 = mix_t0_and_t1

        self.input = dataset_metadata['hparams']['input']

        hdf_paths = Path(f"data/processed/hdf/{dataset_metadata['hparams']['input']}/").glob("*.hdf")
        with open(str(list(hdf_paths)[0]).replace("/hdf/", "/bands/").replace(".hdf",".json"), "r") as f:
            self.bands = json.load(f)

        if channels:
            self.bands = [self.bands[c] for c in channels]

        self.data_type = data_type
        self.image_size = image_size

        if self.image_size:
            self.transform = T.Resize((image_size, image_size))

        self.channels = channels

        self.date_2_hdf_path = {}

        for hdf_path in Path(f"data/processed/hdf/{dataset_metadata['hparams']['input']}").glob("*.hdf"):
            date = str(hdf_path.name).split(".")[0]
            date = "".join(str(hdf_path.name).split(".")[0].split("-")[-2:])
            self.date_2_hdf_path[date] = hdf_path #h5py.File(hdf_path, "r")

        self.normalize_input = normalize_input

        self.use_time_series = use_time_series
        self.max_time_series_length = max_time_series_length

        self.time_series_statistics = time_series_statistics

        self.time_series_folder_name = time_series_folder_name
        self.label_overrides = {}

        if use_label_overrides:
            for path in sorted(Path(label_reviews_folder).glob("*.json")):
                with path.open("r") as f:
                    label_overrides = json.load(f)
                    self.label_overrides.update(label_overrides)
            print(f"label overrides loaded from {label_reviews_folder}")

        self.max_median_cloud_prob = max_median_cloud_prob

        self.drop_repeated_dates = drop_repeated_dates
        self._preloaded_data = None
        self.n_alerts_with_repeated_days = None
        if self.use_time_series and preload_time_series:
            self._preloaded_data = self.load_time_series()

        self.original_paths = self.paths

        self.labels = None
        if preload_labels:
            self.preload_labels()
        self.ignore_images = ignore_images

        self.alert_geometry_dataset = None
        if mask_alert_geometry:
            self.alert_geometry_dataset = HdfDataset(
                dataset_path=dataset_path,
                input_override="closeup-alert_geometry",
                image_size=image_size,
                subset=subset
            )


    def _load_time_series_item(self, path):
        image_id = path.split("/")[-1].split(".")[0]
        date = path.split("/")[-1].split("-")[0]
        path_to_time_series = Path(f"data/processed/{self.time_series_folder_name}/{date}/{image_id}.json")
        with path_to_time_series.open("r") as f:
            time_series = json.load(f)

        array_days = []
        unique_date_indexes = []

        date = str(path).split("/")[-1].split("-")[0]
        for i, ts_date in enumerate(time_series["dates"]):
            int_ts_date = [int(s) for s in ts_date.split("-")]
            d0 = datetime.date(int_ts_date[0], int_ts_date[1], int_ts_date[2])
            df = datetime.date(int(date[:4]), int(date[-2:])%12+1, 1)
            days = (df - d0).days
            array_days.append(days)

            if i==0 or ts_date != time_series["dates"][i-1]:
                unique_date_indexes.append(i)

        if self.n_alerts_with_repeated_days != None:
            self.n_alerts_with_repeated_days += len(unique_date_indexes) != len(time_series["dates"])
        time_series.pop("dates")

        if self.drop_repeated_dates:
            for k1 in time_series.keys():
                for k2 in time_series[k1].keys():
                    time_series[k1][k2] = [time_series[k1][k2][i] for i in unique_date_indexes]
            array_days = [array_days[i] for i in unique_date_indexes]


        if self.max_median_cloud_prob != None:
            cloud_prob_tensor = torch.tensor(time_series["cloud_prob"]["p50"])
            not_cloud = torch.nonzero(cloud_prob_tensor<= self.max_median_cloud_prob).flatten()


        time_series = [time_series[k1][k2] for k1 in time_series for k2 in time_series[k1] if k2 in self.time_series_statistics]
        time_series.append(array_days)
        time_series_tensor = torch.tensor(time_series, dtype=torch.float32)

        if self.max_median_cloud_prob != None:
            time_series_tensor = time_series_tensor[:, not_cloud]

        time_series = time_series_tensor

        if self.max_time_series_length:
            if time_series.shape[1] >= self.max_time_series_length:
                time_series = time_series[:, -self.max_time_series_length:]
            else:
                time_series = torch.nn.functional.pad(time_series, (self.max_time_series_length-time_series.shape[1], 0), "constant", -2)


        return time_series


    def load_time_series(self):
        data = list()
        self.n_alerts_with_repeated_days = 0
        for path in tqdm(self.paths, desc="Loading time series", ncols=DSAD_TQDM_NCOLS):
            data.append(self._load_time_series_item(path))
        data = torch.stack(data)
        print(f"{self.n_alerts_with_repeated_days=}")
        return data


    def __len__(self):
        return len(self.paths)


    def preload_labels(self):
        self.labels = {}
        self.n_pos = 0
        for path in tqdm(self.paths, desc="Loading labels", ncols=DSAD_TQDM_NCOLS):
            image_id = path.split("/")[-1].split(".")[0]
            date = path.split("/")[-1].split("-")[0]
            if image_id in self.label_overrides:
                label = self.label_overrides[image_id]
            else:
                hdf = h5py.File(self.date_2_hdf_path[date], "r")
                label = int(np.array(hdf[image_id]["label"]))
            self.n_pos += label
            self.labels[image_id] = label


    def resample(self, prob_pos=.5, generator=None):
        print("resampling..")
        if not self.labels:
            self.preload_labels()
        pos_weight = prob_pos / self.n_pos
        neg_weight = (1-prob_pos) / (len(self.paths) - self.n_pos)

        sample_weights = torch.tensor(list(self.labels.values()), dtype=int) * pos_weight + (1-torch.tensor(list(self.labels.values()), dtype=int)) *neg_weight
        resampled_indexes = list(torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(self.original_paths), replacement=True, generator=generator))
        self.paths = [self.original_paths[i] for i in resampled_indexes]

        pos_labels = 0
        for path in self.paths:
            image_id = path.split("/")[-1].split(".")[0]
            label = self.labels[image_id]
            pos_labels += label

        print(f"{pos_labels/len(self.paths)=}")


    def __getitem__(self, i):
        path = self.paths[i]
        image_id = path.split("/")[-1].split(".")[0]
        date = path.split("/")[-1].split("-")[0]
        if self.use_time_series:
            if self._preloaded_data is None:
                time_series = self._load_time_series_item(path)
            else:
                time_series = self._preloaded_data[i]

        hdf = h5py.File(self.date_2_hdf_path[date], "r")

        if self.ignore_images:
            image = torch.zeros((1, 1, 1))
        else:
            image_data = np.array(hdf[image_id]["input"], dtype=np.float32)
            image = torch.tensor(image_data, dtype=self.data_type)

            if image.dim() == 3:
                # put channels first
                image = image.permute(2, 0, 1)

                if self.mix_t0_and_t1 and i % 2 == 0:

                    aux = image[:image.shape[0]//2, :, :].detach().clone()
                    image[:image.shape[0]//2, :, :] = image[image.shape[0]//2:, :, :]
                    image[image.shape[0]//2:, :, :] = aux

            elif image.dim() == 2:
                image = image.unsqueeze(0)
            else:
                print(f"error: invalid {image.shape=} (dim = {image.dim()})")
                exit(-1)


            #if self.image_size:
            #    #image = self.transform(image)
            #    image = F.interpolate(input=image.unsqueeze(0), size=self.image_size)
            #    image = image.squeeze()

            if self.image_size:
                image = self.transform(image)

            if self.channels:
                image = image[self.channels, :, :]

            if self.normalize_input:
                image = image / max(torch.max(image), 1)

            if self.alert_geometry_dataset:
                mask = self.alert_geometry_dataset[i][0]
                mask = mask.repeat(image.shape[0], 1, 1)
                image[mask < .5] = 0

        if self.labels:
            label = self.labels[image_id]
        else:
            if image_id in self.label_overrides:
                label = self.label_overrides[image_id]
            else:
                hdf = h5py.File(self.date_2_hdf_path[date], "r")
                label = int(np.array(hdf[image_id]["label"]))

        label = torch.tensor([label], dtype=self.data_type)

        if self.use_time_series:
            # if self.max_time_series_length:
                # time_series = time_series[:, -self.max_time_series_length:]

            return (time_series, image), label, path, i

        else:
            return image, label, path, i

    def show(self, index):

        if self.use_time_series:
            image = self[index][0][1]
            time_series = self[index][0][0]

            print(time_series.shape)
            import matplotlib.pyplot as plt
            plt.cla()
            plt.plot(time_series[0,:])
            plt.xlabel('time')
            plt.ylabel('ndfi-mean')
            plt.show()
        else:
            image = self[index][0]

        band_to_index = {}
        for j, band in enumerate(self.bands):
            band_to_index[band] = j

        rgb = [
            image[[band_to_index["red_t0"], band_to_index["green_t0"], band_to_index["blue_t0"]], :, :].permute(1, 2, 0),
            image[[band_to_index["red_t1"], band_to_index["green_t1"], band_to_index["blue_t1"]], :, :].permute(1, 2, 0),
        ]
        max_rgb = torch.max(torch.max(rgb[0]), torch.max(rgb[1]))
        for i in range(2):
            if max_rgb > 0:
                rgb[i] = rgb[i] / max_rgb
            rgb[i] = np.array(rgb[i] * 255, dtype=np.uint8)

        img = PIL.Image.fromarray(
            np.hstack(
                [
                    rgb[0],
                    rgb[1],
                ]
            )
        )

        scale = 1
        while img.size[0] * scale < 512:
            scale *= 2
        print(f"label = {int(self[index][1])}")
        print(f"id = {self[index][2].split('/')[-1].split('.')[0]}")
        display(img.resize([img.size[0]*scale, img.size[1]*scale], resample=PIL.Image.Resampling.BOX))

class CombineDatasets(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_list,
        time_series_dataset=None,
    ):
        self.dataset_list = dataset_list
        shape = self.dataset_list[0][0][0].shape
        length = len(self.dataset_list[0])
        for dataset in dataset_list:
            assert dataset[0][0].shape[1:] == shape[1:]
            assert len(dataset) == length

        self.time_series_dataset = time_series_dataset
    def __len__(self):
        return len(self.dataset_list[0])

    def __getitem__(self, i):
        X = {}
        X[self.dataset_list[0].input], y, path, i = self.dataset_list[0][i]

        for k in range(1, len(self.dataset_list)):
            X[self.dataset_list[k].input] = self.dataset_list[k][i][0]

        if self.time_series_dataset:
            return (self.time_series_dataset[i][0][0], X), y, path, i
        else:
            return X, y, path, i

    def show(self, index):
        for dataset in self.dataset_list:
            dataset.show(index)

class RotateAndFlip(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
    ):
        self.bands = dataset.bands
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)*8

    def __getitem__(self, i):
        d = i // 8
        r = i % 8

        rotation = r % 4
        translation = r // 4

        if self.dataset.use_time_series:
            (time_series, image), label, path, i = self.dataset[d]
        else:
            image, label, path, i = self.dataset[d]

        if (translation == 1):
            image = T.functional.hflip(image)

        if (rotation > 0):
            image = T.functional.rotate(image, 90*rotation)

        if self.dataset.use_time_series:
            return (time_series, image), label, path, i
        else:
            return image, label, path, i


def shuffled_with_torch_generator(array, generator):
    perm = torch.randperm(len(array), generator=generator)
    return [array[perm[i]] for i in range(len(array))]


class PartiallyShuffledDataset(torch.utils.data.Dataset):
    """Use partial shuffle (see instructions bellow) to significantly speedup training from spinning HDDs.

    Usage: disable shuffle on the dataloader, and call dataset.shuffle(generator) at the end of each epoch.
    """
    def __init__(
        self,
        dataset,
        generator,
        chunk_size=512,
    ):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.shuffled_indexes = list(range(len(self.dataset)))
        self.shuffle(generator)

        if dataset.bands:
            self.bands = dataset.bands

    def shuffle(self, generator):
        list_of_indexes = list(range(len(self.dataset)))

        chunk_size = self.chunk_size
        list_of_chunks = []

        for s in range(0, len(list_of_indexes), chunk_size):
            chunk = list_of_indexes[s : s + chunk_size]
            list_of_chunks.append(chunk)

        list_of_chunks = shuffled_with_torch_generator(list_of_chunks, generator)
        self.shuffled_indexes = []

        for k in range(0, len(list_of_chunks)):
            # torch.randperm(new_indexes, generator=generator)
            new_indexes = shuffled_with_torch_generator(list_of_chunks[k], generator)
            self.shuffled_indexes.extend(new_indexes)

        assert sorted(self.shuffled_indexes) == sorted(list_of_indexes)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        #image_data = tifffile.imread(self.paths[i])
        i = self.shuffled_indexes[i]
        return self.dataset[i]

def tensors_to_device(x, device):
    if isinstance(x, Tuple) or isinstance(x, List):
        for i in range(len(x)):
            x[i] = tensors_to_device(x[i], device)
        return x
    if isinstance(x, Dict):
        for k in x:
            x[k] = tensors_to_device(x[k], device)
        return x

    return x.to(device)

