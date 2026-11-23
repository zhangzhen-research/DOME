import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dome.utils import docid_to_str


class GRDataset(Dataset):
    """(text, doc_idx) pairs -> (input_ids, docid label ids)."""

    def __init__(self, pairs, docid_map, tokenizer, max_len=32, shuffle=False, seed=0):
        self.pairs = [(str(t), int(d)) for t, d in pairs]
        if shuffle:
            random.Random(seed).shuffle(self.pairs)
        self.docid_map = docid_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, doc_idx = self.pairs[i]
        src = self.tokenizer.encode(text, truncation=True, max_length=self.max_len)
        tgt = self.tokenizer.encode(docid_to_str(self.docid_map[doc_idx]))
        return torch.tensor(src), torch.tensor(tgt)

    @staticmethod
    def collate_fn(batch):
        src, tgt = zip(*batch)
        src = pad_sequence(src, batch_first=True, padding_value=0)
        return {
            'input_ids': src,
            'attention_mask': src.ne(0).long(),
            'labels': pad_sequence(tgt, batch_first=True, padding_value=-100),
        }


def build_train_pairs(docs, doc_offset, pseudo, queries):
    """docs: list of doc texts whose global index starts at doc_offset;
    pseudo / queries: list of [text, doc_idx]."""
    pairs = [(d, doc_offset + i) for i, d in enumerate(docs)]
    pairs += [(q, d) for q, d in pseudo]
    pairs += [(q, d) for q, d in queries]
    return pairs
