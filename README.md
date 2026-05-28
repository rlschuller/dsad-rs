# DSAD

## Setup instructions

### Installing pyenv and poetry

Install pyenv with
```sh
curl https://pyenv.run | bash
```

Add 
```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
at the end of *~/.profile* to enable pyenv. 

Install Poetry with
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
Add 
```bash
export PATH="~/.local/bin:$PATH"
```
at the end of *~/.profile* to enable Poetry. 

Restart bash session for the changes to take effect, or run
```sh
. ~/.profile
```

Install python build dependecies
```sh
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

To install Python 3.11 and activate it run
```sh
pyenv install 3.11
pyenv shell 3.11
```

To enable VSCode compatibility run the following command
```sh
poetry config virtualenvs.in-project true
```
Using Poetry install all dependencies.
```sh
poetry install
```

## Source code

All source code is in src/; the subfolders are:

* `src/data`: for downloading data, outputting processed data and creating reports about the data
* `src/models`: for loading the data, training the model and outputting predictions
* `src/eval`: for loading model predictions, outputting model metrics and creating reports about the model
* `src/utils`: for storing auxiliary functions

Note that most of the scripts respond to -h/--help.

### Data pipeline

Scripts that manipulate files in data/.

- `src/data/process-*`: convert data from data/raw/ into more efficient
  formats, and save it in data/processed/. To run all of the scripts, use
  src/data/process-all.sh;

- `src/data/split-*`: create dataset splits from processed files and save the
  results as JSON files in data/processed/datasets/. To run all of the scripts, use
  src/data/split-all.sh;

### Models pipeline

Contains unified train.py and predict.py, and all of the model definitions,
both of which save files into the models/ folder.

### Eval pipeline

Evaluate models with predictions from the models/ folder and save results into
eval/.

See
```
python src/eval/eval.py --help
```
for more details.

### src/utils/

Functions used by more than one script, to be imported from notebooks or other files in src/.


