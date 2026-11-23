import json
import os
import random

import numpy as np
import torch


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path, data, indent=None):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def docid_to_str(codes):
    """[3, 17, 5, 9] -> '$3$$17$$5$$9$'"""
    return '$' + '$$'.join(str(int(c)) for c in codes) + '$'


def str_to_docid(s):
    """'$3$$17$$5$$9$' or '3 17 5 9' -> '3,17,5,9' (canonical key for comparison)."""
    parts = [p for p in s.replace('$', ' ').split() if p]
    return ','.join(parts)


def add_docid_tokens(tokenizer, model, id_num):
    tokenizer.add_tokens([f'${i}$' for i in range(id_num)])
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))


def load_docid_map(paths):
    """Load one or more corpus_docids files ([[doc_idx, [c1..c4]], ...]) into {doc_idx: [codes]}."""
    mapping = {}
    for p in paths:
        for doc_idx, codes in read_json(p):
            mapping[int(doc_idx)] = [int(c) for c in codes]
    return mapping


class Trie:
    """Prefix trie over token id sequences, used for constrained decoding."""

    def __init__(self, sequences=()):
        self.root = {}
        for seq in sequences:
            self.add(seq)

    def add(self, seq):
        node = self.root
        for tok in seq:
            node = node.setdefault(tok, {})

    def get(self, prefix):
        node = self.root
        for tok in prefix:
            node = node.get(tok)
            if node is None:
                return []
        return list(node.keys())


def build_docid_trie(tokenizer, docid_map, start_token_id=0):
    seqs = [[start_token_id] + tokenizer.encode(docid_to_str(c)) for c in docid_map.values()]
    return Trie(seqs)


def eval_metrics(predictions, labels):
    """predictions: list of ranked lists of canonical docids; labels: list of canonical docids."""
    hit1, hit10, hit100, mrr = [], [], [], []
    for preds, label in zip(predictions, labels):
        rank = preds.index(label) + 1 if label in preds else None
        hit1.append(int(rank == 1))
        hit10.append(int(rank is not None and rank <= 10))
        hit100.append(int(rank is not None and rank <= 100))
        mrr.append(1.0 / rank if rank else 0.0)
    return {
        'hit@1': float(np.mean(hit1)),
        'hit@10': float(np.mean(hit10)),
        'hit@100': float(np.mean(hit100)),
        'mrr': float(np.mean(mrr)),
        'num': len(labels),
    }
