import json
import random
import subprocess
from pathlib import Path
import traceback
import time
from src.utils.log import get_log_info

from unique_names_generator import get_random_name

def get_git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_modified_files():
    return (
        subprocess.check_output(["git", "diff", "--name-status"])
        .decode("ascii")
        .strip()
    )

def get_hostname():
    return (
        subprocess.check_output("hostname")
         .decode("ascii")
         .strip()
    )

def get_date():
    return (
        subprocess.check_output(["date", "-Is"]).decode("ascii").strip()
    )


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_deterministic_name(hparams):
    state = random.getstate()

    string_seed = json.dumps(hparams)
    random.seed(string_seed)
    out = get_random_name(separator="_", style="lowercase")

    random.setstate(state)
    return out

def run_notebook_template(template_path, dict_args, out_path):
    assert template_path.endswith(".ipynb"), "template path must be a .ipynb file"
    assert out_path.endswith(".ipynb"), "output path must be a .ipynb file"

    template = Path(template_path).open("r").read()
    for var in dict_args:
        template = template.replace(f"<<{var}>>", dict_args[var])
    out_folder = Path(out_path).parents[0]
    out_folder.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(template)
    subprocess.check_output(['jupyter', 'nbconvert', '--execute', '--inplace', out_path])
    subprocess.check_output(['jupyter', 'nbconvert', '--to', 'html', out_path])

def try_and_log(sleep_seconds, n_times):
    def decorator(func):
        def wrapper_try_and_log(*args, **kwargs):
            j_error = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    if (j_error >= n_times):
                        raise err
                    print(
                        json.dumps(
                            {
                                "function": func.__name__,
                                "error": traceback.format_exc(),
                                "log_info": get_log_info(),
                                "i_error": j_error,
                            }
                        ),
                        flush=True
                    )
                    j_error += 1
                    time.sleep(sleep_seconds)

        return wrapper_try_and_log
    return decorator
