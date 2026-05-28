import argparse
import json

import tifffile
import numpy as np
from tqdm import tqdm


def get_cloud_coverage(img, cloud_threshold=50):
    image_data = np.array(tifffile.imread(img["cloud"]), dtype=np.float32)
    cloud_mask = 1*(image_data > cloud_threshold)
    # print(image_data.dtype, image_data.shape)
    # print(image_data.max())
    coverage = np.sum(cloud_mask)/(image_data.shape[0]*image_data.shape[1])
    soft_coverage = np.sum(image_data)/(image_data.shape[0]*image_data.shape[1])/100

    return cloud_mask, bool((cloud_mask > 0).any()), coverage, soft_coverage


def get_cloud_coverage_stats(img):
    image_data = np.array(tifffile.imread(img["cloud"]), dtype=np.float32)
    image_data = np.clip(image_data, 0, 100)

    return image_data.mean(), image_data.std()


def cloud_on_best_image(data, cloud_threshold, sar_data=None):
    best_images = dict()
    for t in ["t0", "t1"]:
        # has_clouds = list()
        for aid, t0_t1 in tqdm(data.items()):
            if aid not in best_images:
                best_images[aid] = dict({
                    "t0": dict(),
                    "t1": dict(),
                    "label": None,
                })
            has_cloud_l = list()
            # hard_cover_l = list()
            # soft_cover_l = list()

            dates = list(t0_t1[t]["dates"].keys())
            reverse = True if t=="t1" else False
            dates = sorted(dates, reverse=reverse)
            for date in dates:
                _, has_c, hard_cover, soft_cover = get_cloud_coverage(t0_t1[t]["dates"][date], cloud_threshold=cloud_threshold)
                has_cloud_l.append(1*has_c)
                # hard_cover_l.append(hard_cover)
                # soft_cover_l.append(soft_cover)

            min_id = np.argmin(has_cloud_l) ## stable
            best_images[aid][t] = dict({
                "data": dates[min_id],
            })
            if sar_data:
                k = list(sar_data[aid][t].keys())
                if k:
                    sar_date = sorted(list(sar_data[aid][t].keys()), reverse=reverse)[0]
                else:
                    sar_date = None
                best_images[aid][t]["sar_data"] = sar_date

            if t=="t1":
                best_images[aid]["label"] = t0_t1[t]["label"]
        
    return best_images

### Alterar essa funcao para usar sar
def cloud_on_best_image_stats(data, cloud_threshold, cloud_mean, cloud_std, sar_data=None):
    print(f"Using stats {cloud_mean=} {cloud_std=}")
    best_images = dict()
    # for t0 use threshold
    for aid, t0_t1 in tqdm(data.items()):
        if aid not in best_images:
            best_images[aid] = dict({
                "t0": dict(),
                "t1": dict(),
                "label": None,
            })

        has_cloud_l = list()

        dates = list(t0_t1["t0"]["dates"].keys())
        dates = sorted(dates) #ascending order
        for date in dates:
            _, has_c, hard_cover, soft_cover = get_cloud_coverage(t0_t1["t0"]["dates"][date], cloud_threshold=cloud_threshold)
            has_cloud_l.append(1*has_c)
        min_id = np.argmin(has_cloud_l) ## stable
        best_images[aid]["t0"]["data"] = dates[min_id]
        if sar_data:
            k = list(sar_data[aid]["t0"].keys())
            if k:
                sar_date = sorted(list(sar_data[aid]["t0"].keys()))[0]
            else:
                sar_date = None
            best_images[aid]["t0"]["sar_data"] = sar_date
    
    # for t1 use stats
    for aid, t0_t1 in tqdm(data.items()):
        fit_criteria = list()
        dates = list(t0_t1["t1"]["dates"].keys())
        dates = sorted(dates) #ascending order
        for date in dates:
            mean, std = get_cloud_coverage_stats(t0_t1["t1"]["dates"][date])
            if mean < cloud_mean and std < cloud_std:
                fit_criteria.append(date)

        if fit_criteria:
            best_images[aid]["t1"]["data"] = fit_criteria[-1]
        else:
            best_images[aid]["t1"]["data"] = dates[-1]

        best_images[aid]["label"] = t0_t1["t1"]["label"]

        if sar_data:
            k = list(sar_data[aid]["t1"].keys())
            if k:
                sar_date = sorted(list(sar_data[aid]["t1"].keys()), reverse=True)[0]
            else:
                sar_date = None
            best_images[aid]["t1"]["sar_data"] = sar_date
        
    return best_images


def main(args):
    print("Loading index file...", flush=True, end="")
    with open(args.index, "r") as fd:
        data = json.load(fd)
    print("done!")
    
    sar_data = None
    if args.add_sar and args.sar_index:
        print("Loading SAR index file...", flush=True, end="")
        with open(args.sar_index, "r") as fd:
            sar_data = json.load(fd)
        print("done!")

    if args.use_stats is False:
        best_images = cloud_on_best_image(data, args.cloud_threshold, sar_data)
    else:
        best_images = cloud_on_best_image_stats(data, args.cloud_threshold, args.cloud_mean, args.cloud_std, sar_data)
    with open(args.output, "w") as fd:
        json.dump(best_images, fd, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset index to find best t0/t1 pair of images."
    )
    parser.add_argument(
        "--index", required=True, help="Dataset index file."
    )
    parser.add_argument(
        "--add_sar", action="store_true", help="Add S1 GRD SAR layer."
    )
    parser.set_defaults(add_sar=False)
    parser.add_argument(
        "--sar_index", default="data/raw/image_collections/s1_grd-index.json", type=str, help="SAR index."
    )
    parser.add_argument(
        "--cloud_threshold", default=5, type=float, help="Cloud probability threshold."
    )
    parser.add_argument(
        "--use_stats", action='store_true', help="Use stats instead of a threashold."
    )
    parser.set_defaults(use_stats=False)
    parser.add_argument(
        "--cloud_mean", default=5, type=float, help="Cloud probability max mean value."
    )
    parser.add_argument(
        "--cloud_std", default=5, type=float, help="Cloud probability max std value."
    )
    parser.add_argument(
        "--output", default="best_images.json", type=str, help="Output file name."
    )
    args = parser.parse_args()
    print(args)
    main(args)