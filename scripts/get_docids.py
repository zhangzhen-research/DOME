"""Residual-quantization docids.

The RQ codebooks (3 x KMeans(128)) are fit on initial docs only; new docs are encoded
with the frozen codebooks. A 4th digit disambiguates collisions, chosen so that new
docids never clash with existing ones.

Outputs (in --data_dir):
  corpus_docids_rq_1.json  [[doc_idx, [c1, c2, c3, c4]], ...] for initial docs
  corpus_docids_rq_2.json  same for new docs
  rq_model.joblib
"""
import argparse
import os
import random
import sys
from collections import defaultdict

# OpenBLAS crashes ("Bad memory unallocation", "double free") on machines with more
# cores than it was compiled for; cap the thread pools before numpy/sklearn load.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '32')

import joblib
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.utils import read_json, write_json


class ResidualQuantizer:
    def __init__(self, n_layers=3, n_clusters=128, seed=0):
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.seed = seed
        self.codebooks = []

    def fit(self, x):
        residual = x.copy()
        self.codebooks = []
        for layer in range(self.n_layers):
            km = KMeans(n_clusters=self.n_clusters, random_state=self.seed + layer, n_init=1)
            codes = km.fit_predict(residual)
            self.codebooks.append(km.cluster_centers_)
            residual = residual - km.cluster_centers_[codes]
            print(f'RQ layer {layer}: residual norm {np.linalg.norm(residual, axis=1).mean():.4f}')
        return self

    def encode(self, x):
        residual = x.copy()
        codes = []
        for cb in self.codebooks:
            cb_sq = (cb ** 2).sum(1)
            c = np.concatenate([
                (cb_sq[None] - 2 * residual[i:i + 4096] @ cb.T).argmin(1)
                for i in range(0, len(residual), 4096)])
            codes.append(c)
            residual = residual - cb[c]
        return np.stack(codes, 1)


def assign_last_digit(codes, n_clusters, existing, rng):
    """codes: (N, L) array. Returns list of docid tuples with a collision-breaking last digit."""
    groups = defaultdict(list)
    for i, c in enumerate(codes):
        groups[tuple(int(x) for x in c)].append(i)
    out = [None] * len(codes)
    overflow = 0
    for key, idxs in groups.items():
        free = [r for r in range(n_clusters) if key + (r,) not in existing]
        rng.shuffle(free)
        for j, i in enumerate(idxs):
            if j < len(free):
                docid = key + (free[j],)
            else:  # more than n_clusters docs share a prefix; fall back to a duplicate
                overflow += 1
                docid = key + (rng.randrange(n_clusters),)
            out[i] = docid
            existing.add(docid)
    if overflow:
        print(f'warning: {overflow} docs could not get a unique docid')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--n_clusters', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    d = args.data_dir

    vecs = np.load(os.path.join(d, 'corpus_vectors.npy'))
    n_old = len(read_json(os.path.join(d, 'corpus1.json')))
    old, new = vecs[:n_old], vecs[n_old:]
    rng = random.Random(args.seed)

    rq = ResidualQuantizer(args.n_layers, args.n_clusters, args.seed).fit(old)
    joblib.dump(rq.codebooks, os.path.join(d, 'rq_model.joblib'))

    existing = set()
    old_ids = assign_last_digit(rq.encode(old), args.n_clusters, existing, rng)
    write_json(os.path.join(d, 'corpus_docids_rq_1.json'),
               [[i, list(c)] for i, c in enumerate(old_ids)])

    new_ids = assign_last_digit(rq.encode(new), args.n_clusters, existing, rng)
    write_json(os.path.join(d, 'corpus_docids_rq_2.json'),
               [[n_old + i, list(c)] for i, c in enumerate(new_ids)])
    print(f'unique docids: {len(set(old_ids) | set(new_ids))} / {len(old_ids) + len(new_ids)}')


if __name__ == '__main__':
    main()
