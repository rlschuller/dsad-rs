#!/usr/bin/env bash

find data/raw/time_series-all/ -type f | grep 'mean.json$' | parallel --halt-on-error 0 --progress 'python src/data/process-time_series-combine.py -p {} -of "data/processed/time_series"'
#find data/processed/time_series-all -type f > data/processed/time_series-all.find

find data/raw/time_series-2022-all_bands/ -type f | grep 'json$' | parallel --halt-on-error 2 --progress 'python src/data/process-time_series.py -p {} -of "data/processed/time_series-2022-all_bands/"'
#find data/processed/time_series-all_bands -type f > data/processed/time_series-all_bands.find

find data/raw/time_series-forward/ -type f | grep 'mean.json$' | parallel --halt-on-error 0 --progress 'python src/data/process-time_series-combine.py --bands cloud_prob -p {} -of "data/processed/time_series-forward"'
find data/processed/time_series-forward -type f > data/processed/time_series-forward.find
