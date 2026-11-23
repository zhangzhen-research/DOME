"""Edit requests and stage (c): MEMIT-style closed-form updates of the FFN output
projections (wo) of the critical layers."""
import torch

from dome.nethook import encode_batch, get_module
from dome.utils import docid_to_str


@torch.no_grad()
def build_requests(model, tokenizer, items, docid_map, max_len=32, batch_size=64):
    """items: list of (text, doc_idx). Every docid position the model gets wrong under
    teacher forcing becomes a request {text, prefix (gold tokens before it), target}."""
    device = next(model.parameters()).device
    requests, n_correct = [], 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        enc = tokenizer([str(t) for t, _ in batch], return_tensors='pt', padding=True,
                        truncation=True, max_length=max_len).to(device)
        gold = [tokenizer.encode(docid_to_str(docid_map[int(d)]), add_special_tokens=False) for _, d in batch]
        labels = torch.full((len(batch), max(len(g) for g in gold)), -100, dtype=torch.long)
        for j, g in enumerate(gold):
            labels[j, :len(g)] = torch.tensor(g)
        pred = model(**enc, labels=labels.to(device)).logits.argmax(-1).cpu()
        for j, (text, _) in enumerate(batch):
            wrong = [k for k, tok in enumerate(gold[j]) if pred[j, k] != tok]
            n_correct += int(not wrong)
            for k in wrong:
                requests.append({'text': str(text), 'prefix': gold[j][:k], 'target': gold[j][k]})
    return requests, n_correct


@torch.no_grad()
def _layer_stats(model, tokenizer, requests, zs, wo_name, out_name, scale, max_len, batch_size):
    """K K^T and R K^T, accumulated in chunks. K: wo inputs at the last decoder position;
    R: (z - current output of the last critical FFN) * scale."""
    device = next(model.parameters()).device
    KKt = RKt = 0
    rnorm = 0.0
    for i in range(0, len(requests), batch_size):
        reqs = requests[i:i + batch_size]
        enc, dec, last = encode_batch(tokenizer, [r['text'] for r in reqs], [r['prefix'] for r in reqs],
                                      max_len, device, model.config.decoder_start_token_id)
        rows = torch.arange(len(reqs), device=device)
        buf = {}
        h1 = get_module(model, wo_name).register_forward_hook(
            lambda m, inp, out: buf.__setitem__('k', inp[0][rows, last].double()))
        h2 = get_module(model, out_name).register_forward_hook(
            lambda m, inp, out: buf.__setitem__('h', out[rows, last].double()))
        try:
            model(**enc, decoder_input_ids=dec)
        finally:
            h1.remove()
            h2.remove()
        k = buf['k']
        r = (zs[i:i + batch_size].to(device).double() - buf['h']) * scale
        KKt = KKt + k.T @ k
        RKt = RKt + r.T @ k
        rnorm += r.norm(dim=-1).sum().item()
    return KKt, RKt, rnorm / len(requests)


@torch.no_grad()
def apply_dome(model, tokenizer, requests, zs, cov, hparams, max_len=32, verbose=True):
    """Edit the model in place. zs: (N, d_model) targets for the last critical FFN output;
    cov: {wo module name: (d_ff, d_ff) second moment of preserved keys}.

    Layers are updated one after another; each takes 1 / (remaining layers) of the residual
    still measured at the last critical FFN output."""
    device = next(model.parameters()).device
    out_name = hparams.module_tmp.format(hparams.layers[-1])
    for i, layer in enumerate(hparams.layers):
        wo_name = hparams.module_tmp.format(layer) + '.wo'
        wo = get_module(model, wo_name)
        KKt, RKt, rnorm = _layer_stats(model, tokenizer, requests, zs, wo_name, out_name,
                                       1.0 / (len(hparams.layers) - i), max_len, hparams.key_batch_size)
        A = hparams.mom2_update_weight * cov[wo_name].to(device).double() + KKt
        A += hparams.ridge * torch.eye(A.size(0), device=device, dtype=A.dtype)
        upd = torch.linalg.solve(A, RKt.T).T  # (d_model, d_ff); A is symmetric
        wo.weight.add_(upd.to(wo.weight.dtype))
        if verbose:
            print(f'layer {layer}: |resid| {rnorm:.3f} |upd| {upd.norm().item():.3f} '
                  f'|W| {wo.weight.norm().item():.3f}')
    return model


@torch.no_grad()
def request_accuracy(model, tokenizer, requests, max_len=32, batch_size=64):
    """Fraction of requests whose target token is the argmax at the edited position."""
    device = next(model.parameters()).device
    hits = 0
    for i in range(0, len(requests), batch_size):
        reqs = requests[i:i + batch_size]
        enc, dec, last = encode_batch(tokenizer, [r['text'] for r in reqs], [r['prefix'] for r in reqs],
                                      max_len, device, model.config.decoder_start_token_id)
        logits = model(**enc, decoder_input_ids=dec).logits
        pred = logits[torch.arange(len(reqs), device=device), last].argmax(-1)
        hits += int(pred.eq(torch.tensor([r['target'] for r in reqs], device=device)).sum())
    return hits / max(1, len(requests))
