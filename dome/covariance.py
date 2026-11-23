"""Second-moment statistics C = E[k k^T] of the wo-input keys over preserved knowledge
(query -> docid pairs of the initial corpus, teacher forced, all docid positions)."""
import os

import torch
from tqdm import tqdm

from dome.nethook import get_module
from dome.utils import docid_to_str


@torch.no_grad()
def compute_covariance(model, tokenizer, pairs, docid_map, layer_names, n_samples, batch_size, max_len=32):
    device = next(model.parameters()).device
    sums = {n: None for n in layer_names}
    count = 0
    bar = tqdm(total=n_samples, desc='covariance')
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        enc = tokenizer([str(q) for q, _ in batch], return_tensors='pt', padding=True,
                        truncation=True, max_length=max_len).to(device)
        labels = tokenizer([docid_to_str(docid_map[int(d)]) for _, d in batch], return_tensors='pt',
                           padding=True).input_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        mask = labels.ne(-100)
        buf = {}

        def make(name):
            def fn(_, inp, out):
                buf[name] = inp[0][mask].double()
            return fn

        handles = [get_module(model, n).register_forward_hook(make(n)) for n in layer_names]
        try:
            model(**enc, labels=labels)
        finally:
            for h in handles:
                h.remove()
        for n in layer_names:
            k = buf[n]
            s = k.T @ k
            sums[n] = s if sums[n] is None else sums[n] + s
        count += int(mask.sum())
        bar.update(int(mask.sum()))
        if count >= n_samples:
            break
    bar.close()
    return {n: (s / count).float() for n, s in sums.items()}


def get_covariance(model, tokenizer, pairs, docid_map, layer_names, hparams, cache_dir):
    """Load cached statistics or compute and cache them (one file per layer)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = lambda n: os.path.join(cache_dir, f'{n}.mom2_{hparams.mom2_n_samples}.pt')
    missing = [n for n in layer_names if not os.path.exists(path(n))]
    if missing:
        stats = compute_covariance(model, tokenizer, pairs, docid_map, missing,
                                   hparams.mom2_n_samples, hparams.mom2_batch_size)
        for n, c in stats.items():
            torch.save(c.cpu(), path(n))
    return {n: torch.load(path(n)) for n in layer_names}
