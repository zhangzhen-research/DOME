# DOME: Model Editing for New Document Integration in Generative Information Retrieval

Code for the WWW'26 paper *Model Editing for New Document Integration in Generative Information Retrieval* ([arXiv:2603.02773](https://arxiv.org/abs/2603.02773)).

A generative retrieval (GR) model maps a query to a docid. When new documents arrive, the model cannot generate their docids. DOME adds the new documents by editing a few weights of the trained model, without retraining:

1. **Locate.** Average patching finds the decoder FFN layers that store query→docid knowledge. For T5-large these are decoder layers 14–18.
2. **Edit vectors.** The model is run on each new-document input (the document text and its pseudo-queries) with the gold docid prefix. Every docid position it gets wrong becomes an edit request. For each request, a vector δ is optimised and added to the output of the last critical FFN, so that the model emits the gold token. The training label is a hybrid of the original prediction and the target, `(1-λ)·p_orig + λ·onehot`, with λ rising from 0.3 to 1.0.
3. **Update.** The FFN output projections (`wo`) of the critical layers are updated one after another with a closed-form, MEMIT-style solution. A covariance term over the initial documents' query→docid inputs keeps the old knowledge.

## Layout

```
dome/
  hparams.py      DOME hyper-parameters
  compute_z.py    stage 2: edit vectors
  editor.py       edit requests, stage 3: weight update
  covariance.py   second-moment statistics of preserved knowledge (cached)
  nethook.py      forward-hook helpers
  data.py         GR training dataset
  utils.py        json io, docid strings, prefix trie, metrics
scripts/
  split_data.py    split NQ320k into initial and new documents
  get_vectors.py   BERT [CLS] document embeddings
  get_docids.py    residual-quantization docids
  train_gr.py      train the T5 GR model on the initial documents
  evaluate.py      constrained beam search, Hit@1/10/100 and MRR
  locate_layers.py stage 1: average patching over decoder layers
  run_dome.py      stages 2 and 3: edit the model with new documents
run_all.sh         the full pipeline
```

## Setup

```bash
pip install -r requirements.txt
```

Download `t5-large` and `bert-base-uncased` (for example to `models/`) and pass the local paths to `--model_path` / `--bert_path`.

## Data

Put NQ320k in `data/nq320k/`:

| file | format |
|---|---|
| `corpus.json` | `[doc_text, ...]`; the list index is the doc index |
| `train.json`, `dev.json` | `[[query, doc_idx], ...]` |
| `train.json.qg.json` | pseudo-queries, `[[query, doc_idx], ...]` |

`split_data.py` treats the last 10000 documents as new and writes `corpus1/2.json`, `train1/2.json`, `dev1/2.json` and `train1/2.json.qg.json` (1 = initial, 2 = new). Doc indices stay global.

## Running

`bash run_all.sh` runs every step on 8 GPUs; edit the variables at the top first. The steps are:

```bash
DATA=data/nq320k
LAUNCH="accelerate launch --multi_gpu --num_processes 8"

# 1. split, then build docids: RQ codebooks are fit on the initial docs; new docs reuse them
python scripts/split_data.py  --data_dir $DATA --num_new 10000
python scripts/get_vectors.py --data_dir $DATA --bert_path models/bert-base-uncased
python scripts/get_docids.py  --data_dir $DATA

# 2. train the GR model on the initial docs (50 epochs, lr 1e-3, batch 128 per GPU)
$LAUNCH scripts/train_gr.py --data_dir $DATA --model_path models/t5-large --output_dir output/gr-t5-large

# 3. evaluate before editing
$LAUNCH scripts/evaluate.py --data_dir $DATA --model_path output/gr-t5-large/final \
    --splits dev1.json dev2.json

# 4. (optional) locate critical layers
python scripts/locate_layers.py --data_dir $DATA --model_path output/gr-t5-large/final

# 5. edit 100 new docs with DOME
$LAUNCH scripts/run_dome.py --data_dir $DATA --model_path output/gr-t5-large/final \
    --output_dir output/dome-100 --num_docs 100

# 6. evaluate after editing: initial docs (dev1) and the edited new docs
$LAUNCH scripts/evaluate.py --data_dir $DATA --model_path output/dome-100 \
    --splits dev1.json output/dome-100/dev2_edited.json --output output/dome-100/eval.json
```

All scripts also run on one GPU with `python` instead of `accelerate launch`.

### `run_dome.py`

| option | default | meaning |
|---|---|---|
| `--num_docs` | 100 | number of new documents to add |
| `--sample_from` | `dev2` | `dev2`: sample among new docs that have dev2 queries, so every edited doc is evaluated; `all`: any new doc |
| `--pseudo_num` | 10 | pseudo-queries per document added to the edit set |
| `--hparams` | none | JSON file overriding fields of [dome/hparams.py](dome/hparams.py) |
| `--cache_dir` | `output/stats` | covariance cache, computed once per layer from `train1.json` |
| `--z_dir` | `<output_dir>/z` | edit-vector cache |

Outputs in `--output_dir`: the edited model, `edit_config.json` (arguments, hyper-parameters, edited doc indices), and `dev2_edited.json` (the dev2 queries of the edited docs).

Stage 2 is the slow part, and it is split across the processes. Each process saves its edit vectors to `--z_dir`, and process 0 runs stage 3. Saved edit vectors are reused when only stage 3 settings change (`mom2_update_weight`, layers other than the last). Point several runs to the same `--z_dir` to tune those settings quickly:

```bash
echo '{"mom2_update_weight": 1000}' > hp.json
$LAUNCH scripts/run_dome.py --model_path output/gr-t5-large/final --output_dir output/dome-100-w1000 \
    --z_dir output/dome-100/z --hparams hp.json
```

Delete `--cache_dir` and `--z_dir` after retraining the GR model.

### Main hyper-parameters

| name | default | meaning |
|---|---|---|
| `layers` | 14–18 | critical decoder layers |
| `v_num_grad_steps`, `v_lr` | 400, 0.5 | edit-vector optimisation (Adam) |
| `lambda_start`, `lambda_end`, `lambda_steps` | 0.3, 1.0, 50 | hybrid-label schedule |
| `v_min_prob` | 0.9 | optimisation stops only when every target token reaches this probability |
| `mom2_update_weight` | 300 | weight of the preserved-knowledge covariance; higher means less forgetting and weaker edits |
| `mom2_n_samples` | 400000 | docid tokens used for the covariance |
| `z_batch_size` | 256 | requests optimised together; lower it if GPU memory is short |

## Results

NQ320k, T5-large, 8 GPUs, editing 100 new documents that have dev2 queries (101 queries). Default settings.

| model | dev1 Hit@1 | dev1 Hit@10 | edited docs Hit@1 | edited docs Hit@10 | edited docs MRR |
|---|---|---|---|---|---|
| GR before editing | 0.698 | 0.872 | 0.002* | 0.006* | – |
| DOME | 0.681 | 0.863 | 0.723 | 0.762 | 0.741 |

\* measured on all dev2 queries. With 101 queries, edited-doc numbers vary by about ±0.05.

## Citation

```bibtex
@inproceedings{zhang2026dome,
  title     = {Model Editing for New Document Integration in Generative Information Retrieval},
  author    = {Zhang, Zhen and Wang, Zihan and Ma, Xinyu and Wang, Shuaiqiang and Yin, Dawei and
               Xin, Xin and Ren, Pengjie and de Rijke, Maarten and Ren, Zhaochun},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  year      = {2026},
  eprint    = {2603.02773},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR}
}
```
