# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import cky, crf
from parser.utils.common import bos, eos, pad, unk
from parser.utils.corpus import Corpus, Treebank
from parser.utils.field import (BertField, CharField, ChartField, Field,
                                RawField)
from parser.utils.fn import build, factorize
from parser.utils.metric import LabelMetric

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
            self.WORD = Field('words', pad=pad, unk=unk,
                              lower=True)
            self.FEAT = CharField('chars', pad=pad, unk=unk,
                                  fix_len=args.fix_len,
                                  tokenize=list)
            if args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)
            self.LABEL = Field('labels')
            self.fields = Treebank(WORD=(self.WORD, self.FEAT),
                                   LABEL=self.LABEL)

            train = Corpus.load(args.ftrain, self.fields)
            if args.fembed:
                embed = Embedding.load(args.fembed, args.unk)
            else:
                embed = None
            self.WORD.build(train, args.min_freq, embed)
            self.FEAT.build(train)
            self.LABEL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            self.WORD, self.FEAT = self.fields.WORD
            self.LABEL = self.fields.LABEL
        self.criterion = nn.CrossEntropyLoss()

        args.update({
            'n_words': self.WORD.vocab.n_init,
            'n_feats': len(self.FEAT.vocab),
            'n_labels': len(self.LABEL.vocab),
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index
        })

        print(f"Override the default configs\n{args}")
        print(f"{self.WORD}\n{self.FEAT}\n{self.LABEL}")

    def train(self, loader):
        self.model.train()

        for words, feats, labels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.args.pad_index)

            scores = self.model(words, feats)
            loss = self.get_loss(scores, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, LabelMetric()

        for words, feats, labels in loader:
            mask = words.ne(self.args.pad_index)
            lens = mask.sum(1).tolist()
            scores = self.model(words, feats)
            loss = self.get_loss(scores, labels, mask)
            total_loss += loss.item()
            metric(scores.argmax(-1), labels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_labels = []
        for words, feats in loader:
            mask = words.ne(self.args.pad_index)
            scores = self.model(words, feats)
            pred_labels = scores.argmax(-1).tolist()
            all_labels.extend(pred_labels)
        all_labels = [self.LABEL.vocab.id2token(sequence)
                      for sequence in all_labels]

        return all_labels

    def get_loss(self, scores, labels, mask):
        return self.criterion(scores[mask], labels[mask])
