"""Stage (b): optimise an edit vector delta for every request so that the last critical
layer's FFN output z = h + delta makes the model emit the target docid token.

Hybrid label (paper Eq. 9): y = (1 - lambda) * p_orig + lambda * onehot(target),
with lambda annealed linearly from lambda_start to lambda_end over lambda_steps."""
import torch
import torch.nn.functional as F

from dome.nethook import encode_batch, get_module


def compute_z(model, tokenizer, requests, hparams, max_len=32, verbose=True):
    """requests: list of dicts with 'text', 'prefix' (token ids) and 'target' (token id).
    Returns (zs: (N, d_model) float32, success: (N,) bool)."""
    device = next(model.parameters()).device
    module = get_module(model, hparams.module_tmp.format(hparams.layers[-1]))
    for p in model.parameters():
        p.requires_grad_(False)

    zs, ok = [], []
    for b in range(0, len(requests), hparams.z_batch_size):
        reqs = requests[b:b + hparams.z_batch_size]
        enc, dec, last = encode_batch(tokenizer, [r['text'] for r in reqs], [r['prefix'] for r in reqs],
                                      max_len, device, model.config.decoder_start_token_id)
        target = torch.tensor([r['target'] for r in reqs], device=device)
        rows = torch.arange(len(reqs), device=device)
        with torch.no_grad():
            enc_out = model.get_encoder()(**enc)

        state = {'delta': None, 'init': None}

        def edit(_, inp, out):
            if state['init'] is None:
                state['init'] = out[rows, last].detach().float()
            if state['delta'] is None:
                return out
            out = out.clone()
            out[rows, last] = out[rows, last] + state['delta'].to(out.dtype)
            return out

        handle = module.register_forward_hook(edit)
        try:
            def forward():
                logits = model(encoder_outputs=enc_out, attention_mask=enc['attention_mask'],
                               decoder_input_ids=dec).logits
                return logits[rows, last].float()

            with torch.no_grad():
                p_orig = forward().softmax(-1)
            init = state['init']
            delta = torch.zeros_like(init, requires_grad=True)
            state['delta'] = delta
            opt = torch.optim.Adam([delta], lr=hparams.v_lr)
            onehot = F.one_hot(target, p_orig.size(-1)).float()

            for step in range(hparams.v_num_grad_steps):
                frac = min(1.0, step / max(1, hparams.lambda_steps))
                lam = hparams.lambda_start + (hparams.lambda_end - hparams.lambda_start) * frac
                label = (1 - lam) * p_orig + lam * onehot
                logp = forward().log_softmax(-1)
                nll = -(label * logp).sum(-1).mean()
                decay = hparams.v_weight_decay * (delta.norm(dim=-1) / init.norm(dim=-1)).mean()
                loss = nll + decay
                hit = logp.argmax(-1).eq(target)
                p_tgt = logp[rows, target].exp()
                if verbose and step % 20 == 0:
                    print(f'  z batch {b // hparams.z_batch_size} step {step}: nll {nll.item():.4f} '
                          f'acc {hit.float().mean().item():.3f} min p {p_tgt.min().item():.3f} '
                          f'lambda {lam:.2f}')
                if (lam >= hparams.lambda_end and hit.all() and nll.item() < hparams.early_stop_loss
                        and p_tgt.min().item() >= hparams.v_min_prob):
                    break
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    n = delta.norm(dim=-1, keepdim=True)
                    delta.mul_(torch.clamp(hparams.max_delta_norm / (n + 1e-8), max=1.0))

            with torch.no_grad():
                ok.append(forward().argmax(-1).eq(target).cpu())
            zs.append((init + delta).detach())
        finally:
            handle.remove()
    return torch.cat(zs), torch.cat(ok)
