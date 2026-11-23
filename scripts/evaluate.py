"""Constrained beam-search evaluation of a GR model.

The trie covers all docids (initial + new), so the same model is scored on dev1
(queries for initial docs) and dev2 (queries for new docs).
Single GPU: python scripts/evaluate.py ...; several: accelerate launch scripts/evaluate.py ...
"""
import argparse
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.utils import (build_docid_trie, docid_to_str, eval_metrics, load_docid_map,
                        read_json, str_to_docid, write_json)


@torch.no_grad()
def predict(model, tokenizer, queries, docid_map, trie, num_beams=100, batch_size=4, max_len=32,
            disable_bar=False):
    model.eval()
    device = next(model.parameters()).device
    preds, labels = [], []
    for i in tqdm(range(0, len(queries), batch_size), desc='eval', disable=disable_bar):
        batch = queries[i:i + batch_size]
        enc = tokenizer([str(q) for q, _ in batch], return_tensors='pt', padding=True,
                        truncation=True, max_length=max_len).to(device)
        out = model.generate(**enc, max_length=16, num_beams=num_beams, num_return_sequences=num_beams,
                             early_stopping=True,
                             prefix_allowed_tokens_fn=lambda _, sent: trie.get(sent.tolist()))
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for j in range(len(batch)):
            preds.append([str_to_docid(t) for t in texts[j * num_beams:(j + 1) * num_beams]])
        labels += [str_to_docid(docid_to_str(docid_map[int(d)])) for _, d in batch]
    return preds, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--model_path', required=True, help='HF model dir (trained or edited)')
    p.add_argument('--splits', nargs='+', default=['dev1.json', 'dev2.json'],
                   help='file names under --data_dir, or paths')
    p.add_argument('--num_beams', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_queries', type=int, default=None)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    acc = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(acc.device)
    d = args.data_dir
    docid_map = load_docid_map([os.path.join(d, 'corpus_docids_rq_1.json'),
                                os.path.join(d, 'corpus_docids_rq_2.json')])
    trie = build_docid_trie(tokenizer, docid_map, model.config.decoder_start_token_id)

    results = {}
    for split in args.splits:
        queries = read_json(split if os.path.exists(split) else os.path.join(d, split))[:args.max_queries]
        shard = queries[acc.process_index::acc.num_processes]
        preds, labels = predict(model, tokenizer, shard, docid_map, trie, args.num_beams, args.batch_size,
                                disable_bar=not acc.is_main_process)
        preds, labels = gather_object(preds), gather_object(labels)
        if acc.is_main_process:
            results[split] = eval_metrics(preds, labels)
            print(split, results[split])
    if args.output and acc.is_main_process:
        write_json(args.output, results, indent=2)


if __name__ == '__main__':
    main()
