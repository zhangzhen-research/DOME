"""Stage (a): locate critical decoder layers by average patching.

For each decoder layer l, the FFN output at the current decoding position is replaced
by its mean over a sample of query->docid inputs, and we measure how much the
probability of the gold docid token drops. Layers with the largest drop are the ones
that store query->docid knowledge; DOME edits a contiguous window of them.
"""
import argparse
import os
import random
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.nethook import collect_outputs, encode_batch, get_module
from dome.utils import docid_to_str, load_docid_map, read_json, write_json


def make_samples(tokenizer, pairs, docid_map):
    """One sample per (query, docid position): prefix = gold tokens before it."""
    samples = []
    for q, d in pairs:
        gold = tokenizer.encode(docid_to_str(docid_map[int(d)]), add_special_tokens=False)
        for k in range(len(gold)):
            samples.append({'text': str(q), 'prefix': gold[:k], 'target': gold[k]})
    return samples


@torch.no_grad()
def target_prob(model, tokenizer, samples, max_len, batch_size, patch=None):
    """patch: (module, mean vector) replacing the module output at the last position."""
    device = next(model.parameters()).device
    probs = []
    for i in range(0, len(samples), batch_size):
        s = samples[i:i + batch_size]
        enc, dec, last = encode_batch(tokenizer, [x['text'] for x in s], [x['prefix'] for x in s],
                                      max_len, device, model.config.decoder_start_token_id)
        rows = torch.arange(len(s), device=device)
        handle = None
        if patch is not None:
            module, mean = patch

            def fn(_, inp, out):
                out = out.clone()
                out[rows, last] = mean.to(out.dtype)
                return out
            handle = module.register_forward_hook(fn)
        try:
            logits = model(**enc, decoder_input_ids=dec).logits[rows, last].float()
        finally:
            if handle is not None:
                handle.remove()
        tgt = torch.tensor([x['target'] for x in s], device=device)
        probs.append(logits.softmax(-1)[rows, tgt].cpu())
    return torch.cat(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--model_path', required=True)
    p.add_argument('--queries', default='train1.json', help='query file used for patching')
    p.add_argument('--num_queries', type=int, default=500)
    p.add_argument('--module_tmp', default='decoder.block.{}.layer.2.DenseReluDense')
    p.add_argument('--window', type=int, default=5, help='number of consecutive layers to select')
    p.add_argument('--max_len', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--output', default='output/layer_effects.json')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).cuda().eval()
    d = args.data_dir
    docid_map = load_docid_map([os.path.join(d, 'corpus_docids_rq_1.json'),
                                os.path.join(d, 'corpus_docids_rq_2.json')])
    pairs = read_json(os.path.join(d, args.queries))
    pairs = random.Random(args.seed).sample(pairs, min(args.num_queries, len(pairs)))
    samples = make_samples(tokenizer, pairs, docid_map)

    base = target_prob(model, tokenizer, samples, args.max_len, args.batch_size)
    print(f'clean target prob: {base.mean().item():.4f}')
    n_layers = model.config.num_decoder_layers
    names = [args.module_tmp.format(l) for l in range(n_layers)]
    means = collect_outputs(model, tokenizer, [s['text'] for s in samples], [s['prefix'] for s in samples],
                            names, args.max_len, args.batch_size)

    effects = []
    for l, name in enumerate(tqdm(names, desc='patching')):
        patched = target_prob(model, tokenizer, samples, args.max_len, args.batch_size,
                              patch=(get_module(model, name), means[name].mean(0)))
        effects.append((base - patched).mean().item())
        print(f'layer {l}: prob drop {effects[-1]:.4f}')

    w = args.window
    scores = [sum(effects[i:i + w]) for i in range(n_layers - w + 1)]
    start = max(range(len(scores)), key=lambda i: scores[i])
    layers = list(range(start, start + w))
    print(f'selected layers: {layers}')
    write_json(args.output, {'clean_prob': base.mean().item(), 'effects': effects,
                             'selected_layers': layers}, indent=2)


if __name__ == '__main__':
    main()
