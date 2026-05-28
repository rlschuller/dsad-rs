import os
import json
import argparse

import h5py
import tifffile
import numpy as np
from tqdm import tqdm

def to_smaller_size(d0, d1):
    min_dim_0 = min(d0.shape[0], d1.shape[0])
    min_dim_1 = min(d0.shape[1], d1.shape[1])
    d0 = d0[:min_dim_0, :min_dim_1, :]
    d1 = d1[:min_dim_0, :min_dim_1, :]
    return d0, d1

def main(args):
    with open(args.index, "r") as fd:
        data = json.load(fd)

    
    for aid, t0t1 in tqdm(data.items()):
        tmp = aid.split("-")[0]
        year, month = tmp[:4], tmp[4:]
        # create output dir: add year-month subfolder
        out_dir = f"{args.output_path}/{year}-{month}"
        os.makedirs(out_dir, exist_ok=True) 
        # 
        out_filename = f"{out_dir}/{aid}.tif"
        out_bands_filename = f"{out_dir}/{aid}-bands.json"
        label = None
        out_label_filename = f"{out_dir}/{aid}-label.json"

        date_id = t0t1["t0"]["data"]
        t0_file = f"{args.img_path}/{year}-{month}/{aid}/t0/{date_id}.tif"
        t0_tif = tifffile.TiffFile(t0_file)
        t0_data = t0_tif.pages[0].asarray()
        # print("t0 shape", t0_data.shape)

        t0_bands_file = f"{args.img_path}/{year}-{month}/{aid}/t0/{date_id}-bands.json"
        # print(t0_bands_file)
        with open(t0_bands_file, "r") as fd:
            t0_bands = json.load(fd)

        if args.add_sar:
            sar_id = t0t1["t0"]["sar_data"]
            if sar_id is None:
                sar_t0_data = np.zeros((*t0_data.shape[:2], 3), dtype=np.float32)
                sar_t0_bands = ['VV', 'VH', 'angle']
            else:
                sar_t0_file = f"{args.sar_path}/{year}-{month}/{aid}/t0/{sar_id}.tif"
                sar_t0_tif = tifffile.TiffFile(sar_t0_file)
                sar_t0_data = sar_t0_tif.pages[0].asarray()
            # print("SAR shape", sar_t0_data.shape)
                if sar_t0_data.shape[:2] != t0_data.shape[:2]:
                    t0_data, sar_t0_data = to_smaller_size(t0_data, sar_t0_data)
                sar_t0_band_file = f"{args.sar_path}/{year}-{month}/{aid}/t0/{sar_id}-bands.json"
                with open(sar_t0_band_file, "r") as fd:
                    sar_t0_bands = json.load(fd)
            t0_data = np.dstack([t0_data, sar_t0_data])


        date_id = t0t1["t1"]["data"]
        t1_bands_file = f"{args.img_path}/{year}-{month}/{aid}/t1/{date_id}-bands.json"
        with open(t1_bands_file, "r") as fd:
            t1_bands = json.load(fd)

        label = t0t1["label"]

        if args.t1_mosaic:  # use mosaic instead of tiff
            mosaic_file = f"{args.mosaics_path}/{year}-{month}.hdf"
            hdf = h5py.File(mosaic_file, mode="r")
            grp = hdf[aid]
            with open(f"{args.mosaics_bands}/{year}-{month}.json", "r") as fd:
                mosaic_bands = json.load(fd)
                # if {b}_t1 not in mosaic_bands this will throw an error...
                band_index = [mosaic_bands.index(f"{b}_t1") for b in t1_bands ]
            
            t1_data = grp["input"][:, :, band_index]
            # print("t1 shape", t1_data.shape)
            # print("Label:", grp["label"], int(np.array(grp["label"])))

        else:
            t1_file = f"{args.img_path}/{year}-{month}/{aid}/t1/{date_id}.tif"
            t1_tif = tifffile.TiffFile(t1_file)
            t1_data = t1_tif.pages[0].asarray()

        if args.add_sar:
            sar_id = t0t1["t1"]["sar_data"]
            if sar_id is None:
                sar_t1_data = np.zeros((*t1_data.shape[:2], 3), dtype=np.float32)
                sar_t1_bands = ['VV', 'VH', 'angle']
            else:
                sar_t1_file = f"{args.sar_path}/{year}-{month}/{aid}/t1/{sar_id}.tif"
                sar_t1_tif = tifffile.TiffFile(sar_t1_file)
                sar_t1_data = sar_t1_tif.pages[0].asarray()
            # print("SAR shape", sar_t1_data.shape)
                if sar_t1_data.shape[:2] != t1_data.shape[:2]:
                    t1_data, sar_t1_data = to_smaller_size(t1_data, sar_t1_data)
                sar_t1_band_file = f"{args.sar_path}/{year}-{month}/{aid}/t1/{sar_id}-bands.json"
                with open(sar_t1_band_file, "r") as fd:
                    sar_t1_bands = json.load(fd)

            t1_data = np.dstack([t1_data, sar_t1_data])

        if t1_data.shape != t0_data.shape:
            t0_data, t1_data = to_smaller_size(t0_data, t1_data)

        out_data = np.dstack([t0_data, t1_data])
        # print("Out data shape", out_data.shape)
        tifffile.imwrite(out_filename, out_data, photometric='minisblack')

        new_bands = [f"{k}_t0" for k in t0_bands]
        if args.add_sar:
            new_bands += [f"{k}_t0" for k in sar_t0_bands]
        new_bands += [f"{k}_t1" for k in t1_bands]
        if args.add_sar:
            new_bands += [f"{k}_t1" for k in sar_t1_bands]

        with open(out_bands_filename, "w") as fd:
            json.dump(new_bands, fd)            

        with open(out_label_filename, "w") as fd:
            fd.write(f"{label}")            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join t0/t1 best images in one tiff file."
    )
    parser.add_argument(
        "--index", required=True, help="Dataset index file."
    )
    parser.add_argument(
        "--img_path", default="data/raw/image_collections/closeup-copernicus-s2_sr_harmonized", type=str, help="Images root path."
    )
    parser.add_argument(
        "--add_sar", action="store_true", help="Add S1 GRD SAR layer."
    )
    parser.set_defaults(add_sar=False)
    parser.add_argument(
        "--sar_path", default="data/raw/image_collections/closeup-copernicus-s1_grd", type=str, help="SAR root path."
    )
    parser.add_argument(
        "--output_path", default="./", type=str, help="Output folder."
    )
    parser.add_argument(
        "--t1_mosaic", action="store_true", help="Load mosaic for t1 image."
    )
    parser.set_defaults(t1_mosaic=False)
    parser.add_argument(
        "--mosaics_path", default="data/processed/hdf/closeup", type=str, help="Path to mosaics."
    )
    parser.add_argument(
        "--mosaics_bands", default="data/processed/bands/closeup", type=str, help="Path to mosaics band files."
    )

    args = parser.parse_args()
    print(args)
    main(args)