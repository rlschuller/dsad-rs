
TRAIN=$( ./bin/eval/viz/plot -b data/viz/planet/202411 -p models/residualconv/2024-09-11T17:44:24-03:00-olive_bonobo//predictions-tsv/train.tsv -s 150 test.png -e '(tp + fn) / (tp + tn + fp + fn)'  --cmap .05,.3 --print | awk '{print "data/processed/tiffs/" $(NF) ".tif"}' |sort| jq -R | jq -s )
 TEST=$( ./bin/eval/viz/plot -b data/viz/planet/202411 -p models/residualconv/2024-09-11T17:44:24-03:00-olive_bonobo//predictions-tsv/test.tsv  -s 150 test.png -e '(tp + fn) / (tp + tn + fp + fn)'  --cmap .05,.3 --print | awk '{print "data/processed/tiffs/" $(NF) ".tif"}' |sort| jq -R | jq -s )
  VAL=$( ./bin/eval/viz/plot -b data/viz/planet/202411 -p models/residualconv/2024-09-11T17:44:24-03:00-olive_bonobo//predictions-tsv/val.tsv   -s 150 test.png -e '(tp + fn) / (tp + tn + fp + fn)'  --cmap .05,.3 --print | awk '{print "data/processed/tiffs/" $(NF) ".tif"}' |sort| jq -R | jq -s )


mkdir -p data/processed/datasets
( echo "$TRAIN" ; echo "$VAL" ; echo "$TEST" ) | jq -s '{train:.[0], val:.[1], test:.[2], hparams:{name:"spatially-balanced", input:"closeup"}}' > data/processed/datasets/spatially-balanced-residualconv.json

