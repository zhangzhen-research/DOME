"""Integrate new documents into a trained GR model with DOME.

Edit set: for each of --num_docs new documents, the document text plus up to --pseudo_num
pseudo-queries, all mapped to the document's docid.

Stage (b) is split across processes (python ... for one GPU, accelerate launch for several).
Each process saves its edit vectors to --z_dir; process 0 waits for all of them, then runs
stage (c) and saves the edited model. Saved edit vectors are reused when only stage (c)
settings change.
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.compute_z import compute_z
from dome.covariance import get_covariance
from dome.editor import apply_dome, build_requests, request_accuracy
from dome.hparams import DOMEHyperParams
from dome.utils import load_docid_map, read_json, set_seed, write_json

Z_KEYS = ['module_tmp', 'v_lr', 'v_num_grad_steps', 'v_weight_decay', 'max_delta_norm',
          'lambda_start', 'lambda_end', 'lambda_steps', 'early_stop_loss', 'v_min_prob']


def shard_path(z_dir, rank, world):
    return os.path.join(z_dir, f'z_{rank}_of_{world}.pt')


def load_shard(path, meta):
    if not os.path.exists(path):
        return None
    try:
        s = torch.load(path)
    except Exception:  # still being written
        return None
    return s if s['meta'] == meta else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--model_path', required=True, help='trained GR model dir')
    p.add_argument('--output_dir', default='output/dome')
    p.add_argument('--hparams', default=None, help='json file overriding fields of DOMEHyperParams')
    p.add_argument('--num_docs', type=int, default=100, help='number of new docs to integrate')
    p.add_argument('--sample_from', choices=['dev2', 'all'], default='dev2',
                   help="'dev2': only new docs that have dev2 queries; 'all': any new doc")
    p.add_argument('--pseudo_num', type=int, default=10)
    p.add_argument('--max_len', type=int, default=32)
    p.add_argument('--cache_dir', default='output/stats', help='covariance cache')
    p.add_argument('--z_dir', default=None, help='edit-vector cache (default <output_dir>/z)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    set_seed(args.seed)  # every process builds the same request list

    hp = DOMEHyperParams.from_json(args.hparams) if args.hparams else DOMEHyperParams()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).cuda().eval()

    d = args.data_dir
    docid_map = load_docid_map([os.path.join(d, 'corpus_docids_rq_1.json'),
                                os.path.join(d, 'corpus_docids_rq_2.json')])
    n_old = len(read_json(os.path.join(d, 'corpus1.json')))
    new_docs = read_json(os.path.join(d, 'corpus2.json'))
    dev2 = read_json(os.path.join(d, 'dev2.json'))
    pseudo = defaultdict(list)
    for q, idx in read_json(os.path.join(d, 'train2.json.qg.json')):
        pseudo[int(idx)].append(q)

    if args.sample_from == 'dev2':
        pool = sorted({int(x[1]) - n_old for x in dev2})
    else:
        pool = list(range(len(new_docs)))
    doc_ids = sorted(random.sample(pool, min(args.num_docs, len(pool))))
    items = []
    for i in doc_ids:
        g = n_old + i
        items.append((new_docs[i], g))
        items += [(q, g) for q in pseudo[g][:args.pseudo_num]]
    log(f'{len(doc_ids)} new docs, {len(items)} edit inputs, {world} process(es)')

    requests, n_correct = build_requests(model, tokenizer, items, docid_map, args.max_len)
    random.shuffle(requests)
    log(f'{n_correct} inputs already correct; {len(requests)} edit requests')

    # stage (b): edit vectors, split over processes
    meta = {'num_requests': len(requests), 'last_layer': hp.layers[-1],
            'z_hparams': {k: getattr(hp, k) for k in Z_KEYS}, 'model_path': args.model_path,
            'num_docs': args.num_docs, 'sample_from': args.sample_from,
            'pseudo_num': args.pseudo_num, 'seed': args.seed}
    z_dir = args.z_dir or os.path.join(args.output_dir, 'z')
    os.makedirs(z_dir, exist_ok=True)
    my_path = shard_path(z_dir, rank, world)
    if load_shard(my_path, meta) is None:
        t = time.time()
        zs, ok = compute_z(model, tokenizer, requests[rank::world], hp, args.max_len, verbose=main_proc)
        torch.save({'meta': meta, 'zs': zs.cpu(), 'ok': ok}, my_path + '.tmp')
        os.replace(my_path + '.tmp', my_path)
        print(f'[rank {rank}] {len(zs)} edit vectors done in {(time.time() - t) / 60:.1f} min')
    else:
        print(f'[rank {rank}] reusing {my_path}')
    if not main_proc:
        return

    shards = [None] * world
    while True:
        for r in range(world):
            if shards[r] is None:
                shards[r] = load_shard(shard_path(z_dir, r, world), meta)
        missing = [r for r in range(world) if shards[r] is None]
        if not missing:
            break
        print(f'waiting for edit vectors of ranks {missing} ...')
        time.sleep(60)
    zs = torch.empty(len(requests), shards[0]['zs'].size(-1))
    ok = torch.empty(len(requests), dtype=torch.bool)
    for r, s in enumerate(shards):
        zs[r::world], ok[r::world] = s['zs'], s['ok']
    print(f'edit vectors reach target for {ok.float().mean().item():.3f} of {len(requests)} requests')

    # stage (c): preserved-knowledge covariance + layer-wise update
    wo_names = [hp.module_tmp.format(l) + '.wo' for l in hp.layers]
    cov_pairs = read_json(os.path.join(d, 'train1.json'))
    random.Random(args.seed).shuffle(cov_pairs)
    cov = get_covariance(model, tokenizer, cov_pairs, docid_map, wo_names, hp, args.cache_dir)
    apply_dome(model, tokenizer, requests, zs, cov, hp, args.max_len)
    print(f'request accuracy after editing: {request_accuracy(model, tokenizer, requests, args.max_len):.4f}')

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    edited = [n_old + i for i in doc_ids]
    write_json(os.path.join(args.output_dir, 'edit_config.json'),
               {'args': vars(args), 'hparams': hp.to_dict(), 'edited_doc_idx': edited}, indent=2)
    edited = set(edited)
    write_json(os.path.join(args.output_dir, 'dev2_edited.json'), [x for x in dev2 if int(x[1]) in edited])
    print(f'saved edited model to {args.output_dir}')


if __name__ == '__main__':
    main()
