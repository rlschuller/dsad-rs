import argparse
import json
import os

import numpy as np
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, help="path of the tiff file")

parser.add_argument("-b", "--bands", type=str, help="path for the bands file")

args = parser.parse_args()

with open(args.bands, "r") as f:
    bands = json.load(f)

image_id = os.path.basename(os.path.splitext(args.path)[0])

data = tifffile.imread(args.path)

metadata ={}
metadata["id"] = image_id
metadata["variation"] = {}
metadata["max"] = {}
metadata["min"] = {}

for i,band in enumerate(bands):

    metadata["variation"][band] = float(np.max(data[:,:,i])) - float(np.min(data[:,:,i]))
    metadata["max"][band] = float(np.max(data[:,:,i]))
    metadata["min"][band] = float(np.max(data[:,:,i]))

print(json.dumps(metadata))
