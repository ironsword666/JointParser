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

def heatmap(corr, xticklabels, labels, name='matrix'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.set_style('white', {'font.sans-serif': ['SimHei', 'Arial']})
    # Set up the matplotlib figure
    cnt = len(corr)
    f, ax = plt.subplots(nrows=cnt, figsize=(15, 2.4*cnt))
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    # Generate a custom diverging colormap
    cmap = "RdBu"
    # Draw the heatmap with the mask and correct aspect ratio
    for i in range(cnt):
        sns.heatmap(corr[i], cmap=cmap, center=0, ax=ax[i],
                    square=True, linewidths=.5,
                    yticklabels=False, xticklabels=xticklabels[0] if i == cnt-1 or cnt > 5 else False,
                    cbar=False)
        text_i, l = labels[i]
        ax[i].set_title(
            f"{text_i}[{''.join(xticklabels[0][text_i])}] → {l}")
    plt.savefig(f'{name}.png')
    plt.close()



# @torch.enable_grad()
# def evaluate(self, loader):
#     self.model.eval()
#     # self.model.train()

#     total_loss, metric = 0, LabelMetric()

#     for words, feats, labels in loader:
#         mask = words.ne(self.args.pad_index)
#         # replace = '过XX建'
#         # chars[0, 20:22] = torch.tensor([self.CHAR.vocab.stoi[char] for char in replace[1:-1]]).to(chars)
#         # bigram[0, 19:22] = torch.tensor([self.CHAR.vocab.stoi[char] for char in [replace[i:i+2] for i in range(len(replace)-1)]]).to(chars)

#         # chars[0, 20:22] = 1
#         # bigram[0, 19:22] = 1

#         word_text = [self.WORD.vocab.itos[word] for word in words[0]]

#         print([(i, word) for i, word in enumerate(word_text)])

#         words = words.repeat(200, 1)
#         feats = feats.repeat(200, 1, 1)

#         lens = mask.sum(1).tolist()

#         with torch.set_grad_enabled(True):
#             scores, embed = self.model(words, feats)
        
#         check_point = [(i, self.LABEL.vocab.itos[labels[0, i]])
#                         for i in range(lens[0])]
#         check_point_int = list(enumerate(labels[0]))
#         print(check_point_int)
#         embed_grad = [torch.autograd.grad(scores[:, i, labels[0, i]].mean(), 
#                                             embed, 
#                                             retain_graph=True)[0].abs().mean(0).view(len(words[0]), 2, -1).mean(-1).cpu().t() 
#                                             for i, l in check_point_int]
#         heatmap(embed_grad, [word_text],
#                 check_point, 'embed_grad_pip_pos')
#         exit()
#         loss = self.get_loss(scores, labels, mask)
#         total_loss += loss.item()
#         metric(scores.argmax(-1), labels, mask)
#     total_loss /= len(loader)

#     return total_loss, metric