# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import viterbi
from parser.utils.common import pad, unk
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, Field, NGramField
from parser.utils.fn import get_spans
from parser.utils.metric import SpanF1Metric

import torch
import torch.nn as nn
from transformers import BertTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.CHAR = Field('chars', pad=pad, unk=unk,
                              lower=True)
            self.LABEL = Field('labels')

            if args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      tokenize=tokenizer.encode)
                self.fields = CoNLL(CHAR=(self.CHAR, self.FEAT),
                                    LABEL=self.LABEL)
            elif args.feat == 'bigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, lower=True)
                self.fields = CoNLL(CHAR=(self.CHAR, self.BIGRAM),
                                    LABEL=self.LABEL)
            elif args.feat == 'trigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, lower=True)
                self.TRIGRAM = NGramField(
                    'trichar', n=3, pad=pad, unk=unk, lower=True)
                self.fields = CoNLL(CHAR=(self.CHAR,
                                          self.BIGRAM,
                                          self.TRIGRAM),
                                    LABEL=self.LABEL)
            else:
                self.fields = CoNLL(CHAR=self.CHAR,
                                    LABEL=self.LABEL)

            train = Corpus.load(args.ftrain, self.fields)
            embed = Embedding.load(
                'data/tencent.char.200.txt',
                args.unk) if args.embed else None
            self.CHAR.build(train, args.min_freq, embed)
            if hasattr(self, 'FEAT'):
                self.FEAT.build(train)
            if hasattr(self, 'BIGRAM'):
                embed = Embedding.load(
                    'data/tencent.bi.200.txt',
                    args.unk) if args.embed else None
                self.BIGRAM.build(train, args.min_freq,
                                  embed=embed,
                                  dict_file=args.dict_file)
            if hasattr(self, 'TRIGRAM'):
                embed = Embedding.load(
                    'data/tencent.tri.200.txt',
                    args.unk) if args.embed else None
                self.TRIGRAM.build(train, args.min_freq,
                                   embed=embed,
                                   dict_file=args.dict_file)
            self.LABEL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat == 'bert':
                self.CHAR, self.FEAT = self.fields.CHAR
            elif args.feat == 'bigram':
                self.CHAR, self.BIGRAM = self.fields.CHAR
            elif args.feat == 'trigram':
                self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
            else:
                self.CHAR = self.fields.CHAR
            self.LABEL = self.fields.LABEL
        self.criterion = nn.CrossEntropyLoss()
        # [B, E, M, S]
        self.trans = (torch.tensor([1., 0., 0., 1.]).log().to(args.device),
                      torch.tensor([0., 1., 0., 1.]).log().to(args.device),
                      torch.tensor([[0., 1., 1., 0.],
                                    [1., 0., 0., 1.],
                                    [0., 1., 1., 0.],
                                    [1., 0., 0., 1.]]).log().to(args.device))

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'n_labels': len(self.LABEL.vocab),
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index
        })

        vocab = f"{self.CHAR}\n{self.LABEL}\n"
        if hasattr(self, 'FEAT'):
            args.update({
                'n_feats': self.FEAT.vocab.n_init,
            })
            vocab += f"{self.FEAT}\n"
        if hasattr(self, 'BIGRAM'):
            args.update({
                'n_bigrams': self.BIGRAM.vocab.n_init,
            })
            vocab += f"{self.BIGRAM}\n"
        if hasattr(self, 'TRIGRAM'):
            args.update({
                'n_trigrams': self.TRIGRAM.vocab.n_init,
            })
            vocab += f"{self.TRIGRAM}\n"

        print(f"Override the default configs\n{args}")
        print(vocab[:-1])

    def train(self, loader):
        self.model.train()

        for data in loader:
            if self.args.feat == 'bert':
                chars, feats, labels = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, labels = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, labels = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, labels = data
                feed_dict = {"chars": chars}

            self.optimizer.zero_grad()

            mask = chars.ne(self.args.pad_index)
            scores = self.model(feed_dict)

            loss = self.get_loss(scores, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SpanF1Metric()

        for data in loader:
            if self.args.feat == 'bert':
                chars, feats, labels = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, labels = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, labels = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, labels = data
                feed_dict = {"chars": chars}

            mask = chars.ne(self.args.pad_index)
            lens = mask.sum(1).tolist()
            scores = self.model(feed_dict)
            loss = self.get_loss(scores, labels, mask)
            pred_labels = [get_spans(self.LABEL.vocab.id2token(pred.tolist()))
                           for pred in viterbi(self.trans, scores, mask)]
            gold_labels = [get_spans(self.LABEL.vocab.id2token(gold.tolist()))
                           for gold in labels[mask].split(lens)]
            total_loss += loss.item()
            metric(pred_labels, gold_labels)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_labels = []
        for data in loader:
            if self.args.feat == 'bert':
                chars, feats = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, labels = data
                feed_dict = {"chars": chars}
            mask = chars.ne(self.args.pad_index)
            scores = self.model(feed_dict)
            pred_labels = viterbi(self.trans, scores, mask)
            all_labels.extend(pred_labels)
        all_labels = [self.LABEL.vocab.id2token(sequence.tolist())
                      for sequence in all_labels]

        return all_labels

    def get_loss(self, scores, labels, mask):
        return self.criterion(scores[mask], labels[mask])
