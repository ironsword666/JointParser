# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import cky, crf
from parser.utils.common import bos, eos, pad, unk
from parser.utils.corpus import Corpus, Treebank
from parser.utils.field import (BertField, ChartField, Field, NGramField,
                                RawField)
from parser.utils.fn import build, factorize
from parser.utils.metric import BracketMetric

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
            self.TREE = RawField('trees')
            self.CHAR = Field('chars', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
            self.POS = Field('pos')

            self.CHART = ChartField('charts', unk=unk)
            if args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=(self.CHAR, self.FEAT),
                                       POS=self.POS,
                                       CHART=self.CHART)
            elif args.feat == 'bigram':
                self.BIGRAM = NGramField('bichar', n=2, pad=pad, unk=unk,
                                         bos=bos, eos=eos, lower=True)
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=(self.CHAR, self.BIGRAM),
                                       POS=self.POS,
                                       CHART=self.CHART)
            elif args.feat == 'trigram':
                self.BIGRAM = NGramField('bichar', n=2, pad=pad, unk=unk,
                                         bos=bos, eos=eos, lower=True)
                self.TRIGRAM = NGramField('trichar', n=3, pad=pad, unk=unk,
                                          bos=bos, eos=eos, lower=True)
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=(self.CHAR, self.BIGRAM,
                                             self.TRIGRAM),
                                       POS=self.POS,
                                       CHART=self.CHART)
            else:
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=self.CHAR,
                                       POS=self.POS,
                                       CHART=self.CHART)

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
            self.CHART.build(train, 10)
            self.POS.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            self.TREE = self.fields.TREE
            if args.feat == 'bert':
                self.CHAR, self.FEAT = self.fields.CHAR
            elif args.feat == 'bigram':
                self.CHAR, self.BIGRAM = self.fields.CHAR
            elif args.feat == 'trigram':
                self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
            else:
                self.CHAR = self.fields.CHAR
            self.POS = self.fields.POS
            self.CHART = self.fields.CHART
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'n_labels': len(self.CHART.vocab),
            'n_pos_labels': len(self.POS.vocab),
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index,
            'bos_index': self.CHAR.bos_index,
            'eos_index': self.CHAR.eos_index
        })

        vocab = f"{self.TREE}\n{self.CHAR}\n{self.POS}\n{self.CHART}\n"
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
                trees, chars, feats, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                trees, chars, bigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                trees, chars, bigram, trigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                trees, chars, pos, (spans, labels) = data
                feed_dict = {"chars": chars}

            self.optimizer.zero_grad()

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(feed_dict)
            loss, _ = self.get_loss(s_span, s_label, spans, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        coarse_mask = torch.nn.functional.one_hot(torch.tensor([self.CHART.label_cluster(label) for label in self.CHART.vocab.itos]), 3).float().log()

        total_loss = 0
        metric = BracketMetric(self.POS.vocab.stoi.keys())

        for data in loader:
            if self.args.feat == 'bert':
                trees, chars, feats, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                trees, chars, bigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                trees, chars, bigram, trigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                trees, chars, pos, (spans, labels) = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(feed_dict)
            loss, s_span = self.get_loss(s_span, s_label, spans, labels, mask)
            if self.args.constrained_label:
                preds = self.decode(s_span, s_label, mask, coarse_mask.to(device=chars.device))
            else:
                preds = self.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            metric([factorize(tree, self.args.delete, self.args.equal)
                    for tree in preds],
                   [factorize(tree, self.args.delete, self.args.equal)
                    for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        coarse_mask = torch.nn.functional.one_hot(torch.tensor([self.CHART.label_cluster(label) for label in self.CHART.vocab.itos]), 3).float().log()
        all_trees = []
        for data in loader:
            if self.args.feat == 'bert':
                trees, chars, feats, pos = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                trees, chars, bigram, pos = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                trees, chars, bigram, trigram, pos = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                trees, chars, pos = data
                feed_dict = {"chars": chars}
            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label = self.model(feed_dict)
            if self.args.marg:
                s_span = crf(s_span, mask, marg=True)
            if self.args.constrained_label:
                preds = self.decode(s_span, s_label, mask, coarse_mask.to(device=chars.device))
            else:
                preds = self.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            all_trees.extend(preds)

        return all_trees

    def get_loss(self, s_span, s_label, spans, labels, mask):
        span_mask = spans & mask
        span_loss, span_probs = crf(s_span, mask, spans, self.args.marg)
        label_loss = self.criterion(s_label[span_mask], labels[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask, corase=None):
        pred_spans = cky(s_span, mask)
        if corase is None:
            pred_labels = (1 + s_label[..., 1:].argmax(-1)).tolist()
            preds = [[(i, j, labels[i][j]) for i, j in spans]
                    for spans, labels in zip(pred_spans, pred_labels)]
        else:
            bz, lens, _, lsize = s_label.shape
            # print(corase)
            pred_values, pred_labels = (s_label.view(bz, lens, lens, lsize, 1) + corase.view(1, 1, 1, lsize, 3))[..., 1:, :].max(-2)
            # print(pred_values.shape)
            pred_labels += 1
            def dt_func(span):
                i, j, value, label = next(span)
                if j == i+1:
                    corase_label = value[:2].argmax(-1)
                    # print(corase_label)
                    return i, j, value, label, corase_label, []
                else:
                    l_i, l_j, l_v, l_l, l_c, l_t = dt_func(span)
                    r_i, r_j, r_v, r_l, r_c, r_t = dt_func(span)
                    if (l_c > 0) == (r_c > 0):
                        if l_c == 0:
                            corase_label = value[:2].argmax(-1)
                            # print(corase_label)
                        else:
                            corase_label = 2
                    else:
                        # TODO more elegant
                        if j - i > self.args.alpha:
                            corase_label = 2
                            l_c = 1 if l_c == 0 else l_c
                            r_c = 1 if r_c == 0 else r_c
                        else:
                            corase_label = 1
                            l_c = 0 if l_c == 1 else l_c
                            r_c = 0 if r_c == 1 else r_c
                    l_l = l_l[l_c]
                    r_l = r_l[r_c]
                    # print(f"{self.CHART.vocab.itos[l_l]}, {self.CHART.vocab.itos[r_l]} -> {self.CHART.vocab.itos[label[corase_label]]}")
                    # print()
                    return i, j, value, label, corase_label, [(l_i, l_j, l_l)] + l_t + [(r_i, r_j, r_l)] + r_t

            def decode_func(span, val, lab):
                # print(lab)
                i, j, _, label, corase_label, span = dt_func(iter([(i, j, val[i, j], lab[i, j]) for i, j in span]))
                return [(i, j, label[corase_label])] + span

            preds = [decode_func(spans, values, labels)
                        for spans, values, labels in zip(pred_spans, pred_values, pred_labels)]
        return preds
