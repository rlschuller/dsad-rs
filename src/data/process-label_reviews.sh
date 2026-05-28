#!/usr/bin/env bash

find data/label_reviews/ -type f | grep 'tsv$' | parallel --halt-on-error 2 --progress 'python src/data/process-label_reviews.py -p {}'
