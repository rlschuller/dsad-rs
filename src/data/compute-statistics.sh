#!/usr/bin/env bash


FOLDER="$1"

mkdir -p data/processed/statistics/

find "$FOLDER" -type f | grep 'tif$'| parallel --progress 'gdalinfo -stats -json {} | jq -c' > "data/processed/statistics/$( basename "$FOLDER" ).jsonl"
