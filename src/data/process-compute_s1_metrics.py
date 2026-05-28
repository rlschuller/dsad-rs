import json
import argparse
from pathlib import Path


import h5py
import numpy as np
from tqdm import tqdm


def rvi(vv, vh):
    return np.sqrt(vv/(vv+vh)*(vv/vh))


def mrvi(vv, vh):
    return np.sqrt(vv/(vv+vh))*(4*vh/(vv+vh))


def mrdi(vv, vh):
    return (vv-vh)/(vv + vh)

def main(args):
    output_folder = Path(args.hdf_output)
    if output_folder.exists() is False:
        output_folder.mkdir(parents=True)

    bands_output_folder = Path(args.bands_output)
    if bands_output_folder.exists() is False:
        bands_output_folder.mkdir(parents=True)

    for hdf in tqdm(list(Path(args.input).glob("*.hdf"))):
        output_file = output_folder / hdf.name
        band_file = bands_output_folder / f"{hdf.stem}.json"
        with h5py.File(str(hdf), "r") as input_hdf, \
             h5py.File(str(output_file), "w") as output_hdf, \
             open(band_file, "w") as bfd:
            for aid in tqdm(input_hdf.keys(), leave=False):
                grp = output_hdf.create_group(aid)
                img_HWC = np.array(input_hdf[aid]["input"])
                # convert data
                img_CHW = np.transpose(img_HWC, axes=(2, 0, 1))
                vv_0 = np.power(10, img_CHW[0]/10)
                vh_0 = np.power(10, img_CHW[1]/10)
                vv_1 = np.power(10, img_CHW[3]/10)
                vh_1 = np.power(10, img_CHW[4]/10)
                img = np.array([
                    vv_0, vh_0,
                    rvi(vv_0, vh_0),
                    mrvi(vv_0, vh_0),
                    mrdi(vv_0, vh_0),
                    vv_1, vh_1,
                    rvi(vv_1, vh_1),
                    mrvi(vv_1, vh_1),
                    mrdi(vv_1, vh_1),
                    vv_0/vv_1,
                    vh_1/vv_0,
                ])
                img = np.transpose(img, axes=(1, 2, 0))
                # print(img.shape)
                #
                grp.create_dataset("input", data=img, compression="lzf")
                grp["label"] = int(np.array(input_hdf[aid]["label"]))
            json.dump(
               ["VV_t0", "VH_t0", "RVI_t0", "MRVI_t0", "MRDI_t0"] +
               ["VV_t1", "VH_t1", "RVI_t1", "MRVI_t1", "MRDI_t1", "DIV01", "DIV10"],
               bfd,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute S1 metrics for HDF files."
    )
    parser.add_argument(
        "--input", default="data/processed/hdf/closeup-copernicus-s1_grd-first_and_last/", type=str, help="Input HDFs folder path",
    )
    parser.add_argument(
        "--hdf_output", required=True, type=str, help="HDF output folder path."
    )
    parser.add_argument(
        "--bands_output", required=True, type=str, help="Bands output folder path."
    )
    args = parser.parse_args()
    main(args)
