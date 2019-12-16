# -*- coding: utf-8 -*-

import os
from parser.utils.common import bos, eos, pad, unk
from parser.utils.corpus import Corpus, Treebank
from parser.utils.field import BertField, CharField, Field, LabelField
from parser.utils.metric import F1Method

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
                              bos=bos, eos=eos, lower=True)
            if args.feat == 'char':
                self.FEAT = CharField('chars', pad=pad, unk=unk,
                                      bos=bos, eos=eos, fix_len=args.fix_len,
                                      tokenize=list)
            elif args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert', pad='[PAD]',
                                      bos='[CLS]', eos='[SEP]',
                                      tokenize=tokenizer.encode)
            else:
                self.FEAT = Field('tags', bos=bos, eos=eos)
            self.LABEL = LabelField('labels')
            if args.feat in ('char', 'bert'):
                self.fields = Treebank(WORD=(self.WORD, self.FEAT),
                                       LABEL=self.LABEL)
            else:
                self.fields = Treebank(WORD=self.WORD, POS=self.FEAT,
                                       LABEL=self.LABEL)

            train = Corpus.load(args.ftrain, self.fields)
            self.WORD.build(train, args.min_freq)
            self.FEAT.build(train)
            self.LABEL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat in ('char', 'bert'):
                self.WORD, self.FEAT = self.fields.WORD
            else:
                self.WORD, self.FEAT = self.fields.WORD, self.fields.POS
            self.LABEL = self.fields.LABEL
        self.criterion = nn.CrossEntropyLoss()

        print(f"{self.WORD}\n{self.FEAT}\n{self.LABEL}")
        args.update({
            'n_words': self.WORD.vocab.n_init,
            'n_feats': len(self.FEAT.vocab),
            'n_labels': len(self.LABEL.vocab),
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index,
            'bos_index': self.WORD.bos_index,
            'eos_index': self.WORD.eos_index,
            'nul_index': self.LABEL.vocab[()]
        })

    def train(self, loader):
        self.model.train()

        total_loss, metric = 0, F1Method(self.args.nul_index)

        for words, feats, labels in loader:
            self.optimizer.zero_grad()

            mask = labels.ne(self.args.pad_index)
            scores = self.model(words, feats)
            loss = self.get_loss(scores, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            preds = self.decode(scores, mask)
            total_loss += loss.item()
            metric(preds, labels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, F1Method(self.args.nul_index)

        for words, feats, labels in loader:
            mask = labels.ne(self.args.pad_index)
            scores = self.model(words, feats)
            loss = self.get_loss(scores, labels, mask)
            preds = self.decode(scores, mask)
            total_loss += loss.item()
            metric(preds, labels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_labels, all_probs = [], [], []
        for words, feats in loader:
            mask = labels.ne(self.args.pad_index)
            lens = mask.sum(1).tolist()
            scores = self.model(words, feats)
            scores = scores.softmax(-1)
            preds = self.decode(scores, mask)
            all_labels.extend(preds[mask].split(lens))
            if self.args.prob:
                arc_probs = scores.gather(-1, preds.unsqueeze(-1))
                all_probs.extend(arc_probs.squeeze(-1)[mask].split(lens))
        all_labels = [seq.tolist() for seq in all_labels]
        all_probs = [[round(p, 4) for p in seq.tolist()] for seq in all_probs]

        return all_labels, all_probs

    def get_loss(self, scores, labels, mask):
        scores, labels = scores[mask], labels[mask]
        loss = self.criterion(scores, labels)

        return loss

    def decode(self, scores, mask):
        return scores.argmax(-1)
