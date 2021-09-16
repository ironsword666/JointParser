# -*- coding: utf-8 -*-

import argparse
import os
from collections import Counter
from parser.config import Config

import torch
from nltk.tree import Tree


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class BracketMetric(Metric):

    def __init__(self, wiki_mode=False, eps=1e-8):
        super(BracketMetric, self).__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps
        self.wiki_mode = wiki_mode

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([split for split, label in pred])
            ugold = Counter([split for split, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        if self.wiki_mode:
            # s = f"| {self.lp * 100 :5.2f} || {self.lr * 100 :5.2f} || {self.lf * 100 :5.2f}"
            s = f", {self.lf * 100 :5.2f}\n"
        else:
            s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} \n"
            s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} \n"
            s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"
        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


def load_trees(file):
    with open(file, 'r') as f:
        trees = [Tree.fromstring(string) for string in f]
    return trees


def factorize(tree, delete_labels=None, equal_labels=None, only_border=True):
    def track(tree, i):
        label = tree.label()
        if delete_labels is not None and label in delete_labels:
            label = None
        if equal_labels is not None:
            label = equal_labels.get(label, label)
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return ((i+len(tree[0]) if label is not None else i), [])
        split, spans = [i], []
        j = i
        for child in tree:
            j, s = track(child, j)
            split += [j]
            spans += s
        if label is not None and j > i:
            if only_border:
                spans = [((i, j), label)] + spans
            else:
                spans = [(tuple(split), label)] + spans
        return j, spans
    return track(tree, 0)[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    parser.add_argument('--conf', '-c', default='config.ini',
                        help='path to config file')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    parser.add_argument('--pred', '-p', required=True, type=str,
                        help='path to config file')
    parser.add_argument('--gold', '-g', required=True, type=str,
                        help='path to saved files')
    parser.add_argument('--detail', '-d', action='store_true',
                        help='show evaluate detail')
    parser.add_argument('--strict', '-s', action='store_true',
                        help='strict mode')
    parser.add_argument('--wiki', '-w', action='store_true',
                        help='output wiki type')
    args = parser.parse_args()
    args = Config(args.conf).update(vars(args))
 
    metric = BracketMetric(wiki_mode=args.wiki)

    preds = load_trees(args.pred)

    trees = load_trees(args.gold)

    metric([factorize(tree, args.delete, args.equal, only_border=not args.strict)
            for tree in preds],
           [factorize(tree, args.delete, args.equal, only_border=not args.strict)
            for tree in trees])
    print(metric)
