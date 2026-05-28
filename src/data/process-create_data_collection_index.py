import json
import argparse
from pathlib import Path
from tqdm import tqdm


def main(args):
    index = dict()
    for ym in tqdm(Path(args.data_path).iterdir()):
        for aid in ym.iterdir():
            if aid.is_dir():
                id = str(aid).split("/")[-1]
                index[id] = dict()
                for t in ["t0", "t1"]:
                    index[id][t] = dict({
                        "dates": dict(),
                        "label": None,
                        "bands": None,
                    })
                    for date in (aid/t).glob("*.tif"):
                        # print(date)
                        d = str(date).split("/")[-1]
                        b = str(date).split(".")[0]
                        cloud_path = Path(f"{args.cloud_path}/{ym.stem}/{id}/{t}/{d}")
                        if cloud_path.exists():
                            if index[id][t]["bands"] is None:
                                with open(f"{b}-bands.json") as fd:
                                    bands = json.load(fd)
                                    index[id][t]["bands"] = bands
                            if index[id][t]["label"] is None:
                                with open(f"{b}-label.json") as fd:
                                    label = json.load(fd)
                                    index[id][t]["label"] = label
                            # print(d)
                            index[id][t]["dates"][d.split(".")[0]] = dict({
                                "data": str(date),
                                "cloud": str(cloud_path),
                            })

    with open(args.output, "w") as fd:
        json.dump(index, fd, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset index to find best t0/t1 pair of images."
    )
    parser.add_argument(
        "--data_path", default="data/raw/image_collections/closeup-copernicus-s2_sr_harmonized", type=str, help="Collection path",
    )
    parser.add_argument(
        "--cloud_path", default="data/raw/image_collections/closeup-copernicus-s2_cloud_probability", type=str, help="Collection path",
    )
    parser.add_argument(
        "--output", default="index.json", type=str, help="Output file name."
    )
    args = parser.parse_args()
    main(args)

