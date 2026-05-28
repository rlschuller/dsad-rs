#!/usr/bin/env bash

echo "$1"

find "$1" | grep 'tif$' |\
    parallel --progress --halt-on-error 2 'printf "%s\t"   {}; gdalinfo {} 2>/dev/null | grep Origin | awk -F"(" "{print \$2}" | sed "s;,;\t;g" | sed "s;);;g"' | \
    awk -F'/' '{print $NF}' | \
    sed 's;\.tif;;'  > \
    data/processed/id-coordinates.tsv

