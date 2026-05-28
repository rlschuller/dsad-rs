import json
from pathlib import Path


LABEL_REVIEWS_FOLDER = "data/processed/label_reviews/"

def load_labels(dataset="data/processed/metadata/closeup", use_label_overrides=True):
    label_overrides = dict()
    if use_label_overrides:
        for path in sorted(Path(LABEL_REVIEWS_FOLDER).glob("*.json")):
            with path.open("r") as f:
                labels = json.load(f)
                label_overrides.update(labels)
    path = Path(dataset)
    if path.exists():
        labels = dict()
        for f in path.glob("*.json"):
            with open(str(f), "r") as fd:
                data = json.load(fd)
                for k, v in data.items():
                    if use_label_overrides and k in label_overrides:
                        labels[k] = label_overrides[k]
                    else:
                        labels[k] = v["label"]
        return labels
    else:
        raise Exception(f"{str(path)} not found")
