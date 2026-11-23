"""Encode every document with BERT [CLS] embeddings -> corpus_vectors.npy (N x 768, global doc order)."""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.utils import read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--bert_path', default='bert-base-uncased')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_len', type=int, default=500)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertModel.from_pretrained(args.bert_path).to(device).eval()

    corpus = read_json(os.path.join(args.data_dir, 'corpus1.json')) + \
        read_json(os.path.join(args.data_dir, 'corpus2.json'))
    vecs = []
    for i in tqdm(range(0, len(corpus), args.batch_size)):
        batch = [str(x) for x in corpus[i:i + args.batch_size]]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True,
                           max_length=args.max_len).to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0, :]
        vecs.append(out.float().cpu().numpy())
    vecs = np.concatenate(vecs, 0)
    np.save(os.path.join(args.data_dir, 'corpus_vectors.npy'), vecs)
    print('saved', vecs.shape)


if __name__ == '__main__':
    main()
