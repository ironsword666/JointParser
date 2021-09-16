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


class SpanF1Metric(Metric):

    def __init__(self, wiki_mode=False, eps=1e-8):
        super(SpanF1Metric, self).__init__()

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
            lpred = Counter(pred)
            lgold = Counter(gold)
            utp = list((upred & ugold).elements())
            ltp = list((lpred & lgold).elements())
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        if self.wiki_mode:
            # return f"| {self.up * 100 :5.2f} || {self.ur * 100 :5.2f} || {self.uf * 100 :5.2f}\n" + f"| {self.lp * 100 :5.2f} || {self.lr * 100 :5.2f} || {self.lf * 100 :5.2f}"
            return f"{self.uf * 100 :5.2f}, {self.lf * 100 :5.2f}"
        else:
            return f"SEG P: {self.up:6.2%} R: {self.ur:6.2%} F: {self.uf:6.2%}\nPOS P: {self.lp:6.2%} R: {self.lr:6.2%} F: {self.lf:6.2%}"

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


def load_poss(file):
    poss = []
    with open(file, 'r') as f:
        buf = []
        for string in f:
            split = string.split()
            if len(split) != 2:
                if len(buf) > 0:
                    poss.append(buf)
                    buf = []
            else:
                buf.append(tuple(split))
    if len(buf) > 0:
        poss.append(buf)
    return poss


def get_spans(labels):
    spans = []
    i = 0
    for label in labels:
        width = len(label[0])
        spans.append(((i, i+width-1), label[1]))
        i += width
    return spans


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
    parser.add_argument('--tree', '-t', action='store_true',
                        help='show evaluate detail')
    parser.add_argument('--wiki', '-w', action='store_true',
                        help='output wiki type')
    args = parser.parse_args()
    args = Config(args.conf).update(vars(args))

    metric = SpanF1Metric(wiki_mode=args.wiki)

    if args.tree:
        trees = load_trees(args.pred)
        preds = [get_spans(tree.pos()) for tree in trees]
    else:
        preds = [get_spans(pos) for pos in load_poss(args.pred)]

    trees = load_trees(args.gold)
    golds = [get_spans(tree.pos()) for tree in trees]

    metric(preds, golds)

    if args.wiki:
        print(metric, end='')
    else:
        print(metric)
