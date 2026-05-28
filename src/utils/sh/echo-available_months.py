import argparse
from src.utils.download import YEAR_AND_MONTH_TO_URLS, AVAILABLE_YEARS

parse = argparse.ArgumentParser(
    description = ("Given a year, echo on the screel all of the months with alerts available from imazon. "
                   "This sh utility facilitates paralelization of the download process, as it feeds bash variables that are used by the GNU-parallel calls.")
)

parse.add_argument(
    "-y",
    "--year",
    type=int,
)

args = parse.parse_args()

assert args.year in AVAILABLE_YEARS, f"{args.year} not in {AVAILABLE_YEARS=}."

for (y, m) in sorted(YEAR_AND_MONTH_TO_URLS):
    if y == args.year:
        print(m)
