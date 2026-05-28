import os

DSAD_TQDM_NCOLS = None

if ("DSAD_TQDM_NCOLS" in os.environ and
    os.environ["DSAD_TQDM_NCOLS"].isnumeric() and
    int(os.environ["DSAD_TQDM_NCOLS"]) > 0):

    DSAD_TQDM_NCOLS = int(os.environ["DSAD_TQDM_NCOLS"])
