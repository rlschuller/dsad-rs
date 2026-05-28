import json
import argparse
from pathlib import Path


def main(args):
    index = dict()
    for ym in Path(args.path).iterdir():
        for aid in ym.iterdir():
            if aid.is_dir():
                id = str(aid).split("/")[-1]
                index[id] = dict()
                for t in ["t0", "t1"]:
                    index[id][t] = dict()
                    for date in (aid/t).glob("*.tif"):
                        b = str(date).split(".")[0]
                        with open(f"{b}-bands.json") as fd:
                            bands = json.load(fd)

                        index[id][t][date.stem] = dict({
                            "data": str(date),
                            "bands": bands,
                        })

    with open(args.output, "w") as fd:
        json.dump(index, fd, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset index to find best t0/t1 pair of images."
    )
    parser.add_argument(
        "--path", default="data/raw/image_collections/closeup-copernicus-s1_grd", type=str, help="Collection path",
    )
    parser.add_argument(
        "--output", default="collection_index.json", type=str, help="Output file name."
    )
    args = parser.parse_args()
    main(args)

