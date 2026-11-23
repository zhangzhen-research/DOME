"""Train the T5 generative retrieval model on initial documents D0
(doc text, pseudo-queries and training queries -> docid)."""
import argparse
import os
import sys

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.data import GRDataset, build_train_pairs
from dome.utils import add_docid_tokens, load_docid_map, read_json, set_seed, write_json


def lr_lambda(warmup, total, min_ratio):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(min_ratio, 1.0 - (step - warmup) / max(1, total - warmup))
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--model_path', default='t5-large')
    p.add_argument('--output_dir', default='output/gr-t5-large')
    p.add_argument('--id_num', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup_ratio', type=float, default=0.2)
    p.add_argument('--min_lr_ratio', type=float, default=5e-4)
    p.add_argument('--max_len', type=int, default=32)
    p.add_argument('--save_interval', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    acc = Accelerator()
    if acc.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        write_json(os.path.join(args.output_dir, 'args.json'), vars(args), indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    add_docid_tokens(tokenizer, model, args.id_num)

    d = args.data_dir
    docid_map = load_docid_map([os.path.join(d, 'corpus_docids_rq_1.json')])
    pairs = build_train_pairs(read_json(os.path.join(d, 'corpus1.json')), 0,
                              read_json(os.path.join(d, 'train1.json.qg.json')),
                              read_json(os.path.join(d, 'train1.json')))
    dataset = GRDataset(pairs, docid_map, tokenizer, args.max_len)
    acc.print(f'training pairs: {len(dataset)}')
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=GRDataset.collate_fn, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    model, optimizer, loader = acc.prepare(model, optimizer, loader)
    total = len(loader) * args.epochs
    scheduler = LambdaLR(optimizer, lr_lambda(int(args.warmup_ratio * total), total, args.min_lr_ratio))

    for epoch in range(args.epochs):
        model.train()
        losses = []
        bar = tqdm(loader, disable=not acc.is_local_main_process, desc=f'epoch {epoch}')
        for batch in bar:
            loss = model(**batch).loss
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            bar.set_postfix(loss=sum(losses) / len(losses))
        acc.wait_for_everyone()
        if acc.is_main_process:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(f'epoch {epoch}: loss {sum(losses) / len(losses):.4f}\n')
            if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
                save_dir = os.path.join(args.output_dir, 'final' if epoch + 1 == args.epochs else f'epoch{epoch}')
                acc.unwrap_model(model).save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    main()
