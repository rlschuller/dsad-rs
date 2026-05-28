import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils.general import get_deterministic_name, get_hostname, get_git_revision, get_date
import sys
import itertools

parser = argparse.ArgumentParser("given a list of model runs or a tag, fit a logistic regression on the 'predictions-without_sigmoid'")

parser.add_argument(
    "-t",
    "--tag",
    default='regression-all_images-no_normalization-residualconv',
    type=str,
    help="select all model runs that match this tag. to use mulitple tags, separate them with commas.",
)


parser.add_argument(
    "--train_fold",
    type=str,
    default="train",
)

parser.add_argument(
    "-ei",
    "--exclude_images",
    type=str,
    nargs="*",
    default=[],
)

parser.add_argument(
    "--test_fold",
    type=str,
    default="val",
)

parser.add_argument(
    "-pf", "--personal_folder", action="store_true", help="use personal folder instead of shared folder"
)

args = parser.parse_args()

MODELS_FOLDER = Path("models/")

hparam_files = sorted(list(str(p) for p in MODELS_FOLDER.glob("**/hparams.json")))

output = {}

#print(hparam_files)

selected_paths = []

tags = args.tag.strip().split(",")

for p in hparam_files:
    with Path(p).open("r") as f:
        hparams = json.load(f)

    if "tag" in hparams and (hparams["tag"] in tags):

        skip = False
        for img in args.exclude_images:
            if "input_override" in hparams and hparams["input_override"] == img:
                skip=True

        #if "input_override" in hparams and (hparams["input_override"] is not None) and (hparams["input_override"] != "closeup-copernicus-s1_grd-first_and_last"):
        #    skip=True

        if not skip:
            selected_paths.append('/'.join(p.split('/')[:-1]))
output["selected_paths"] = selected_paths

# verify consistency of predictions
fold_2_predictions = {}

for fold in ["train", "val", "test"]:
    with Path(f"{selected_paths[0]}/predictions-without_sigmoid/{fold}.jsonl").open("r") as f:
        predictions = f.readlines()
    predictions = [json.loads(x) for x in predictions]
    fold_2_predictions[fold] = predictions

path_2_fold_2_predictions = {}

for p in selected_paths:
    path_2_fold_2_predictions[p] = {}

    for fold in ["train", "val", "test"]:
        with Path(f"{p}/predictions-without_sigmoid/{fold}.jsonl").open("r") as f:
            predictions = f.readlines()
        predictions = [json.loads(x) for x in predictions]

        path_2_fold_2_predictions[p][fold] = predictions

        assert [x["path"] for x in predictions] == [x["path"] for x in fold_2_predictions[fold]]
        assert [x["y"] for x in predictions] == [x["y"] for x in fold_2_predictions[fold]]


y = []
X = []

y_test = []
X_test = []
for i in range(len(fold_2_predictions[args.train_fold])):
    new_line = []
    for p in path_2_fold_2_predictions:
        new_line.append(path_2_fold_2_predictions[p][args.train_fold][i]["pred"])
    X.append(new_line)
    y.append(fold_2_predictions[args.train_fold][i]["y"])

for i in range(len(fold_2_predictions[args.test_fold])):
    new_line = []
    for p in path_2_fold_2_predictions:
        new_line.append(path_2_fold_2_predictions[p][args.test_fold][i]["pred"])
    X_test.append(new_line)
    y_test.append(fold_2_predictions[args.test_fold][i]["y"])

X = np.array(X)
y = np.array(y)

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"{X[:10]=}")
print(f"{y[:10]=}")


from sklearn.svm import l1_min_c

#cs = l1_min_c(X, y, loss="log") * np.logspace(0, 6, 32)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty=None,
        tol=1e-6,
        max_iter=int(1e7),
        warm_start=True,
        fit_intercept=True,
    ),
)


feature_names = []
for i in range(len(selected_paths)):
    with Path(f"{selected_paths[i]}/hparams.json").open("r") as f:
        hparams = json.load(f)
    feature_names.append(f"{hparams['model']}-{hparams['input_override']}-lr={hparams['learning_rate']} ({selected_paths[i]})")

n_features = len(selected_paths)
vacc = []
for comb in itertools.combinations(list(range(len(selected_paths))), 3):
    print(comb)
    clf.fit(X[:, comb], y)
    preds_test = clf.predict(X_test[:,comb])
    accuracy_test = np.mean((preds_test > .5) == y_test)
    coefs = clf["logisticregression"].coef_.ravel().copy()
    vacc.append((accuracy_test, [feature_names[i] for i in comb], coefs))

vacc = sorted(vacc, key=lambda x : x[0])

for x in vacc:
    print(x)

print(f"{n_features=}")
exit(0)
coefs_ = []
acc_ = []
for c in cs:
    clf.set_params(logisticregression__C=c)
    clf.fit(X, y)
    coefs_.append(clf["logisticregression"].coef_.ravel().copy())

    preds_test = clf.predict(X_test)
    accuracy_test = np.mean((preds_test > .5) == y_test)
    acc_.append(accuracy_test)

coefs_ = np.array(coefs_)
print(coefs_)
print(acc_)

feature_names = []
for i in range(len(selected_paths)):
    with Path(f"{selected_paths[i]}/hparams.json").open("r") as f:
        hparams = json.load(f)
    feature_names.append(f"{hparams['model']}-{hparams['input_override']}")

import matplotlib.pyplot as plt

## Colorblind-friendly palette (IBM Color Blind Safe palette)
#colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]

plt.figure(figsize=(20, 12))
for i in range(coefs_.shape[1]):
    plt.semilogx(cs, coefs_[:, i], marker="o", label=feature_names[i])

plt.semilogx(cs, acc_[:], marker="o", label="acc")
ymin, ymax = plt.ylim()
plt.xlabel("C")
plt.ylabel("Coefficients")
plt.title("Logistic Regression Path")
plt.legend()
plt.axis("tight")
plt.savefig('selection.png')


exit(0)



print(f"{cs=}")
reg = LogisticRegression().fit(X, y)
print(f"{reg.coef_=}")
print(f"{reg.intercept_=}")

output["coef"] = reg.coef_.tolist()[0]
output["intercept"] = reg.intercept_.tolist()[0]

preds = reg.predict(X)
preds_test = reg.predict(X_test)

accuracy = np.mean((preds > .5) == y)
accuracy_test = np.mean((preds_test > .5) == y_test)

print(f"{accuracy=}")
print(f"{accuracy_test=}")
print()
coef_2_str = {}
coefs = reg.coef_[0]

for i in range(len(selected_paths)):
    with Path(f"{selected_paths[i]}/hparams.json").open("r") as f:
        hparams = json.load(f)

    if "input_override" not in hparams:
        hparams["input_override"] = ""

    if hparams["input_override"] == None:
        hparams["input_override"] = "closeup"

    coef_2_str[float(reg.coef_[0][i])] = f"{hparams['model']:10s} {hparams['input_override']:40s} {reg.coef_[0][i]:20.15f} {selected_paths[i]}".replace("  ", " ")
    #print(f"{hparams['model']:10s} {hparams['input_override']:40s} {reg.coef_[0][i]:20.15f} {selected_paths[i]}")

coef_2_str_keys = sorted(coef_2_str)
coef_2_str_keys.reverse()
for c in coef_2_str_keys:
    print(coef_2_str[c])



print(json.dumps(output,indent=2))

exit(0)

date_start = get_date()
run_name = get_deterministic_name(output)
base_folder = "personal" if args.personal_folder else "."
output_folder = Path(f"{base_folder}/models/logistic_regression/{run_name}")

output_folder.mkdir(exist_ok=True, parents=True)

with (output_folder / "params.json").open("w") as f:
    json.dump(output, f, indent=2)

info = {}
with (output_folder / "info.json").open("w") as f:
    info["hostname"] = get_hostname()
    info["git"] = get_git_revision()
    info["date_start"] = date_start
    info["argv"] = sys.argv
    f.write(json.dumps(info, indent=4)+"\n")

print(f"{output_folder=}")
exit(0)

# show interesting predictions
path_to_pred_lines = {}
for path in selected_paths:
    with Path(f"{path}/predictions-without_sigmoid/test.jsonl").open("r") as f:
        path_to_pred_lines[path] = f.readlines()

for i in range(len(path_to_pred_lines[selected_paths[0]])):
    preds = []
    for path in path_to_pred_lines:
        pred = json.loads(path_to_pred_lines[path][i])["pred"]
        preds.append(pred)

    tif_path = json.loads(path_to_pred_lines[selected_paths[0]][i])["path"]
    y = json.loads(path_to_pred_lines[selected_paths[0]][i])["y"]
    logistic_pred = np.sum(preds * coefs) + reg.intercept_

    if np.sign(preds * coefs)[0] != np.sign(preds * coefs)[4] and (preds * coefs)[0] < -.5 and y == 0:
        print()
        print()
        print(path_to_pred_lines.keys())
        print(tif_path)
        print(f"{y=}")
        print(f"{logistic_pred=}")
        print(np.array(preds * coefs))
        for j in range(len(selected_paths)):
            with Path(f"{selected_paths[j]}/hparams.json").open("r") as f:
                hparams = json.load(f)

            if hparams["input_override"] == None:
                hparams["input_override"] = "closeup"

            print(f"{hparams['model']:10s} {hparams['input_override']:40s} {(preds * coefs)[j]} {selected_paths[j]}")

