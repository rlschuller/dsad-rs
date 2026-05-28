import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    help="time series directory",
    default="data/raw/time_series-forward/"
)


args = parser.parse_args()

while args.path.endswith("/"):
    args.path = args.path[:-1]

payload_paths = [str(x) for x in Path(args.path).glob("**/*.payload")]



problem_payloads = []

for payload_path in payload_paths:

    with Path(payload_path).open('r') as f:
        payload = json.load(f)

    try:
        for key in payload.keys():
            with Path(payload_path.replace(".payload", f"-{key}.json")).open('r') as f:
                _data = json.load(f)
    except:
        problem_payloads.append(payload_path)

with Path(args.path + ".problems").open("w") as f:
    json.dump(problem_payloads, f)

for payload_path in problem_payloads:
    with Path(payload_path).open('r') as f:
        payload = json.load(f)

    for key in payload.keys():
        path =  Path(payload_path.replace(".payload", f"-{key}.json"))
        print(f"removing {path}..")
        path.unlink(missing_ok=True)
