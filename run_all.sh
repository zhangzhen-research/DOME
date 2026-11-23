#!/usr/bin/env bash
# Full DOME pipeline on NQ320k. Run from the repository root: bash run_all.sh
set -euo pipefail

DATA=data/nq320k                # corpus.json, train.json, dev.json, train.json.qg.json
T5=models/t5-large              # HF name or local path
BERT=models/bert-base-uncased   # HF name or local path
GPUS=8
GR=output/gr-t5-large
NUM_DOCS=100
EDIT=output/dome-$NUM_DOCS
LAUNCH="accelerate launch --multi_gpu --num_processes $GPUS"

# 1. split: the last 10000 docs are new
python scripts/split_data.py --data_dir $DATA --num_new 10000

# 2. docids
python scripts/get_vectors.py --data_dir $DATA --bert_path $BERT
python scripts/get_docids.py  --data_dir $DATA

# 3. train the GR model on initial docs
$LAUNCH scripts/train_gr.py --data_dir $DATA --model_path $T5 --output_dir $GR

# 4. before editing
$LAUNCH scripts/evaluate.py --data_dir $DATA --model_path $GR/final \
    --splits dev1.json dev2.json --output $GR/eval.json

# 5. (optional) locate critical layers; T5-large gives 14-18, the default in dome/hparams.py
python scripts/locate_layers.py --data_dir $DATA --model_path $GR/final --output $GR/layer_effects.json

# 6. DOME
$LAUNCH scripts/run_dome.py --data_dir $DATA --model_path $GR/final --output_dir $EDIT \
    --num_docs $NUM_DOCS --cache_dir $GR/stats

# 7. after editing: initial docs, edited new docs
$LAUNCH scripts/evaluate.py --data_dir $DATA --model_path $EDIT \
    --splits dev1.json $EDIT/dev2_edited.json --output $EDIT/eval.json
