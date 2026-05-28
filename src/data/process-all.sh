#!/usr/bin/env bash

if [ $# -ne 0 ] ; then
    echo "usage: process-all.sh"
    echo ""
    echo "Process all of the files from 'data/raw/', and save the results in 'data/processed/."
    exit 0
fi

find -L data/raw -mindepth 3 -maxdepth 3 | sed 's;data/raw/;;' | parallel -j1 --progress --halt-on-error 2 'python src/data/process-alerts.py -f {} > out/process_{#}.out'

python src/data/process-cell_geometry.py

# avoid training in data that's beeing downloaded
cp -r data/raw/time_series data/processed

