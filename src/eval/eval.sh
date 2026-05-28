#!/usr/bin/env bash


if [ -z "$1" ] ; then
    echo "evaluate all runs of a model family"
    echo "src/eval/eval.sh <model_name> [<model_to_be_compared>]"
    exit
fi

HPARAM_FILES="$( find "models/$1" | grep hparams.json )"

if [ -z "$2" ] ; then
    COMMANDS="$( echo "$HPARAM_FILES" | parallel -k 'echo python src/eval/eval.py -p "$( dirname {} )"' )"
else
    COMMANDS="$( echo "$HPARAM_FILES" | parallel -k "echo python src/eval/eval.py -p \"\$( dirname {} )\" --compare_with $2" )"
fi
echo "Total number of jobs: $( echo "$COMMANDS" | wc -l )"
echo "$COMMANDS" | parallel --progress '{} &> /dev/null'
echo "Concatenating results to 'eval/$1/all_eval.jsonl'..."
find "eval/$1" | grep 'test.json' | parallel -j1 '(cat {} ;cat {//}/val.json) | jq -cs "."' | pv > eval/$1/all_eval.jsonl.tmp
cat eval/$1/all_eval.jsonl.tmp | jq -sc 'sort_by(.[1]."f-measure")[] | .[0] |  (.log.model_log |= -1) + {"out":.log.model_log.output_folder}' > eval/$1/all_eval.jsonl
rm eval/$1/all_eval.jsonl.tmp
