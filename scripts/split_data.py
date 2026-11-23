"""Split NQ320k into initial documents D0 and new documents D' (the last --num_new docs).

Doc indices stay global: doc i in corpus2.json has global index len(corpus1) + i.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dome.utils import read_json, write_json


def split(items, boundary):
    old = [x for x in items if int(x[1]) < boundary]
    new = [x for x in items if int(x[1]) >= boundary]
    return old, new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/nq320k')
    p.add_argument('--num_new', type=int, default=10000)
    args = p.parse_args()
    d = args.data_dir

    corpus = read_json(os.path.join(d, 'corpus.json'))
    boundary = len(corpus) - args.num_new
    write_json(os.path.join(d, 'corpus1.json'), corpus[:boundary])
    write_json(os.path.join(d, 'corpus2.json'), corpus[boundary:])
    print(f'corpus: {boundary} initial / {args.num_new} new docs')

    for name, out1, out2 in [('train.json', 'train1.json', 'train2.json'),
                             ('dev.json', 'dev1.json', 'dev2.json'),
                             ('train.json.qg.json', 'train1.json.qg.json', 'train2.json.qg.json')]:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            print(f'skip missing {path}')
            continue
        old, new = split(read_json(path), boundary)
        write_json(os.path.join(d, out1), old)
        write_json(os.path.join(d, out2), new)
        print(f'{name}: {len(old)} old / {len(new)} new')


if __name__ == '__main__':
    main()
