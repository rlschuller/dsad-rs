import json
import argparse


def main(args):
    print(f"Reading {args.input}. ", end='', flush=True)
    with open(args.input, "r") as fd:
        dataset = json.load(fd)
    print("Done.")

    print(f"Writing {args.output}. ", end='', flush=True)
    with open(args.output, "w") as fd:
        json.dump(dataset, fd, indent=args.level)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-l",
        "--level",
        default=4,
        type=int,
    )
    args = parser.parse_args()
    main(args)