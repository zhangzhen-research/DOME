import torch


def get_module(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def encode_batch(tokenizer, texts, prefixes, max_len, device, start_id=0):
    """Encoder inputs for `texts` and right-padded decoder inputs [start] + prefix.

    Returns enc dict, decoder_input_ids (B, T) and the index of the last real decoder position."""
    enc = tokenizer([str(t) for t in texts], return_tensors='pt', padding=True,
                    truncation=True, max_length=max_len).to(device)
    seqs = [[start_id] + list(p) for p in prefixes]
    dec = torch.full((len(seqs), max(len(s) for s in seqs)), tokenizer.pad_token_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        dec[i, :len(s)] = torch.tensor(s)
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    return enc, dec.to(device), last


@torch.no_grad()
def collect_outputs(model, tokenizer, texts, prefixes, module_names, max_len, batch_size):
    """Outputs of each module at the last decoder position: {name: (N, dim) float32}."""
    device = next(model.parameters()).device
    res = {n: [] for n in module_names}
    for i in range(0, len(texts), batch_size):
        enc, dec, last = encode_batch(tokenizer, texts[i:i + batch_size], prefixes[i:i + batch_size],
                                      max_len, device, model.config.decoder_start_token_id)
        rows = torch.arange(len(last), device=device)
        buf = {}

        def make(name):
            def fn(_, inp, out):
                buf[name] = out[rows, last].float()
            return fn

        handles = [get_module(model, n).register_forward_hook(make(n)) for n in module_names]
        try:
            model(**enc, decoder_input_ids=dec)
        finally:
            for h in handles:
                h.remove()
        for n in module_names:
            res[n].append(buf[n])
    return {n: torch.cat(v) for n, v in res.items()}
