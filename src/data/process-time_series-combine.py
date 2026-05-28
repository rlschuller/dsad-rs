import json
import argparse
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    default="data/raw/time_series-all/202301/202301-00000000000000000000-mean.json",
    help="path to a json time series file",
)

parser.add_argument(
    "-of",
    "--output_folder",
    default="data/processed/time_series/",
    help="path to a json time series file",
)

parser.add_argument(
    "--bands",
    default="ndfi,cloud_prob",
    help="used for filtering: if any of the values is missing, the entry is discarded",
)

args = parser.parse_args()


#paths = list(Path("-".join(args.path.split("-")[:-1])).glob("-*")) # not working, probably thinks  Path("-".join(args.path.split("-")[:-1])) is a folder...

preamble_path = "-".join(args.path.split("-")[:-1])
paths = sorted(glob(preamble_path+"-*.json"))


output_path = Path(args.output_folder) / Path(preamble_path).parts[-2] / (Path(preamble_path).parts[-1] + ".json")

output_path.parent.mkdir(parents=True,exist_ok=True)


out_data = {}


bands = args.bands.split(",")


for path in paths:
    if "dates.json" in path:
        with Path(path).open("r") as f:
            dates = json.load(f)

        out_data["dates"] = dates

    if "mean.json" in path:
        with Path(path).open("r") as f:
            data = json.load(f)

        #for p in feature["properties"]:
        for feature in data["features"]:
            for p in bands:
                if p not in out_data:
                    out_data[p] = {"mean":[]}

                if "mean" not in out_data[p]:
                    out_data[p]["mean"] = []

                if p in feature["properties"]:
                    out_data[p]["mean"].append(feature["properties"][p])
                else:
                    out_data[p]["mean"].append(None)

    if "stddev.json" in path:
        with Path(path).open("r") as f:
            data = json.load(f)

        #for p in feature["properties"]:
        for feature in data["features"]:
            for p in bands:
                if p not in out_data:
                    out_data[p] = {"stddev":[]}

                if "stddev" not in out_data[p]:
                    out_data[p]["stddev"] = []

                if p in feature["properties"]:
                    out_data[p]["stddev"].append(feature["properties"][p])
                else:
                    out_data[p]["stddev"].append(None)

    if "percentile.json" in path:
        with Path(path).open("r") as f:
            data = json.load(f)

        for feature in data["features"]:


            #for p in feature["properties"]:
            for p in [f"{b}_{q}" for b in bands for q in ["p25", "p50", "p75"]]:

                quantile = p.split("_")[-1]
                prop_name = "_".join(p.split("_")[:-1])

                if prop_name not in out_data:
                    out_data[prop_name] = {quantile:[]}

                if quantile not in out_data[prop_name]:
                    out_data[prop_name][quantile] = []

                if p in feature["properties"]:
                    out_data[prop_name][quantile].append(feature["properties"][p])
                else:
                    out_data[prop_name][quantile].append(None)


for k in out_data["cloud_prob"]:

    for b in bands:
        assert len(out_data["dates"]) == len(out_data[b][k]), f"{len(out_data['dates'])} == {len(out_data[b][k])}"


filtered_out_data = {}


filtered_out_data["dates"] = []

for prop_name in bands:
    filtered_out_data[prop_name] = {}

    for stat_name in ["mean", "stddev", "p25", "p50", "p75"]:
        filtered_out_data[prop_name][stat_name] = []

for i in range(len(out_data["dates"])):

    use_i = True
    for prop_name in bands:
        for stat_name in ["mean", "stddev", "p25", "p50", "p75"]:

            if out_data[prop_name][stat_name][i] == None:
                use_i = False


    if use_i:
        filtered_out_data["dates"].append(out_data["dates"][i])
        for prop_name in bands:
            for stat_name in ["mean", "stddev", "p25", "p50", "p75"]:
                filtered_out_data[prop_name][stat_name].append(out_data[prop_name][stat_name][i])


out_data = filtered_out_data
with output_path.open("w") as f:
    json.dump(out_data,f)
