import json
from pathlib import Path

with Path("data/processed/datasets/closeup-2769685.json").open("r") as f:
    dataset = json.load(f)


# to generate, run
# find data/processed/time_series/ | grep json > processed-time_series.find
time_series_find = Path.open("processed-time_series.find").readlines()

available_ids = set()

for line in time_series_find:
    id = line.split("/")[-1].split(".")[0]

    available_ids.add(id)

for split in ["train", "val", "test"]:
    id_list = [
        path for path in dataset[split] if path.split("/")[-1].split(".")[0] in available_ids
    ]
    dataset[split] = id_list

dataset["hparams"]["filter"] = "filter-based_on_time_series"

print(dataset["hparams"])
print(dataset.keys())

size = len(dataset["train"]) + len(dataset["val"]) + len(dataset["test"])
with Path(f"data/processed/datasets/time_series-closeup-{size}.json").open("w") as f:
    json.dump(dataset, f)
