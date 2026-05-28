import json
import argparse


def main(args):
    print(f"Reading {args.dataset1}. ", end='', flush=True)
    with open(args.dataset1, "r") as fd:
        dataset1 = json.load(fd)
    print("Done.")
    print(f"Reading {args.dataset2}. ", end='', flush=True)
    with open(args.dataset2, "r") as fd:
        dataset2 = json.load(fd)
    print("Done.")

    output = dict()
    output["hparams"] = dataset1["hparams"]
    del dataset1["hparams"]

    for fold in dataset1:
        fold_1 = dataset1[fold]
        fold_2 = set(dataset2[fold])
        output[fold] = [a for a in fold_1 if a in fold_2]

    n = len(output["train"]) + len(output["val"]) + len(output["test"])
    print(f"Size of new dataset is {n}.")

    print(f"Writing {args.output}. ", end='', flush=True)
    with open(args.output, "w") as fd:
        json.dump(output, fd, indent=4)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter dataset 1 to a subset of dataset 2."
    )
    parser.add_argument(
        "-1",
        "--dataset1",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-2",
        "--dataset2",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    main(args)