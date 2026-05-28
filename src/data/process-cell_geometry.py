import json
from pathlib import Path

import numpy as np


def get_start_and_stride(grid):
    list_x = []
    list_y = []

    for polygon in grid["features"]:
        coordinates = polygon["geometry"]["coordinates"]

        # for some reason, the list of pairs is nested in a list of a single item.
        # the next assert checks if this polygon indeed follows this unusual convention.
        assert len(coordinates) == 1

        coordinates = coordinates[0]

        min_x = 9999
        min_y = 9999

        for pair in coordinates:
            pair[0] = float(f"{pair[0]:0.3f}")
            pair[1] = float(f"{pair[1]:0.3f}")

            min_x = min(pair[0], min_x)
            min_y = min(pair[1], min_y)

            list_x.append(min_x)
            list_y.append(min_y)

    list_x = np.sort(np.unique(list_x))
    list_y = np.sort(np.unique(list_y))

    dist_x = np.ediff1d(list_x)
    dist_y = np.ediff1d(list_y)

    stride_x = dist_x.max()
    stride_y = dist_y.max()

    start_x = list_x[dist_x.argmax()]
    while start_x > list_x[0]:
        start_x -= stride_x

    start_y = list_y[dist_y.argmax()]
    while start_y > list_y[0]:
        start_y -= stride_y

    start = [start_x, start_y]
    stride = [stride_x, stride_y]

    return start, stride


grid = json.load(Path("data/meta/grid.json").open("r"))
start, stride = get_start_and_stride(grid)

coords_to_name = {}


for polygon in grid["features"]:
    name = polygon["properties"]["grid"]

    coordinates = polygon["geometry"]["coordinates"]

    # for some reason, the list of pairs is nested in a list of a single item.
    # the next assert checks if this polygon indeed follows this unusual convention.
    assert len(coordinates) == 1

    coordinates = coordinates[0]

    min_x = 9999
    min_y = 9999

    for pair in coordinates:
        pair[0] = float(f"{pair[0]:0.3f}")
        pair[1] = float(f"{pair[1]:0.3f}")

        min_x = min(pair[0], min_x)
        min_y = min(pair[1], min_y)

    coord_x = int((min_x - start[0]) // stride[0])
    coord_y = int((min_y - start[1]) // stride[1])

    assert str((coord_x, coord_y)) not in coords_to_name
    coords_to_name[str((coord_x, coord_y))] = name


coords_to_name = {c: coords_to_name[c] for c in sorted(coords_to_name, key=lambda x: (eval(x)[1], eval(x)[0]))}

json.dump(
    {
        "start": start,
        "stride": stride,
        "coords_to_name": coords_to_name,
    },
    Path("data/processed/grid.json").open("w"),
    indent=2,
)
