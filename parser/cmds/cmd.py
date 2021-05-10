# -*- coding: utf-8 -*-

import os

from numpy import argmax
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
        # build fields
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            # create fields
            self.TREE = RawField('trees')
            self.CHAR = Field('chars', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
            self.POS = Field('pos')
            # (i, j, l)
            self.CHART = ChartField('charts', unk=unk)
            # feature engine
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
            # load training corpus
            train = Corpus.load(args.ftrain, self.fields)
            # load pretrained embeddings 
            embed = Embedding.load(
                'data/tencent.char.200.txt',
                args.unk) if args.embed else None
            # initialize fields
            # fields only contain vocabularies but not containing data
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
            # 10 is for low frequency projection
            self.CHART.build(train, 10)
            self.POS.build(train)
            torch.save(self.fields, args.fields)
        else:
            # load fields 
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
        # loss function 
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # update number of chars and labels, identity of specail tokens
        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'n_labels': len(self.CHART.vocab),
            'n_pos_labels': len(self.POS.vocab),
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index,
            'bos_index': self.CHAR.bos_index,
            'eos_index': self.CHAR.eos_index
        })

        # display fields
        vocab = f"{self.TREE}\n{self.CHAR}\n{self.POS}\n{self.CHART}\n"
        # update input dim
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
            # fenceposts length: (B)
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            # for a sentence: seq_len=10, fenceposts=6, pad=2
            # [[True,  True,  True,  True,  True,  True,  True, False, False]]
            # (B, 1, seq_len-1)
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # NOTE: mask.new_ones(seq_len-1, seq_len-1).triu_(1) get a matrix whose upper triangular part is True (discard diagonal)
            # such as:
            # [[False,  True,  True,  True,  True,  True],
            # [False, False,  True,  True,  True,  True],
            # [False, False, False,  True,  True,  True],
            # [False, False, False, False,  True,  True],
            # [False, False, False, False, False,  True],
            # [False, False, False, False, False, False]]
            # NOTE: mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1) get actual part of charts
            # [[False,  True,  True,  True,  True,  True,  True, False, False],
            #  [False, False,  True,  True,  True,  True,  True, False, False],
            #  [False, False, False,  True,  True,  True,  True, False, False],
            #  [False, False, False, False,  True,  True,  True, False, False],
            #  [False, False, False, False, False,  True,  True, False, False],
            #  [False, False, False, False, False, False,  True, False, False],
            #  [False, False, False, False, False, False, False, False, False],
            #  [False, False, False, False, False, False, False, False, False],
            #  [False, False, False, False, False, False, False, False, False]]
            # (B, seq_len-1, seq_len-1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            # (B, seq_len-1, seq_len-1), (B, seq_len-1, seq_len-1, n_labels)
            s_span, s_label = self.model(feed_dict)
            # crf-loss + cross-entropy-loss
            loss, _ = self.get_loss(s_span, s_label, spans, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        # before float and log
        # [[1, 0, 0],
        # [0, 1, 0],
        # [0, 0, 1],
        # [1, 0, 0],
        # [0, 1, 0]]
        # indicate which class the label is 
        # (itos_size, 3), 3 is for POS*, POS, SYN/SYN*
        # after log:
        # [[-inf, -inf, 0.],
        # [-inf, 0., -inf],
        # [-inf, -inf, 0.],
        # [-inf, 0., -inf],
        # [-inf, -inf, 0.],
        # log is for used to mask labels to three kinds of sub-labels
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
            # mbr 
            loss, s_span = self.get_loss(s_span, s_label, spans, labels, mask)
            # TODO no need constrained labeling for evaluate
            if self.args.constrained_label:
                preds = self.decode(s_span, s_label, mask, coarse_mask.to(device=chars.device))
            else:
                preds = self.decode(s_span, s_label, mask)
            # build predicted tree
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            # compare predict results and ground-truth
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
        span_mask = (spans > 0) & mask
        span_loss, span_probs = crf(s_span, mask, self.args.marg)
        label_loss = self.criterion(s_label[span_mask], labels[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask, corase=None):
        pred_spans = cky(s_span, mask)
        if corase is None:
            # [chart: [[]], ]
            pred_labels = (1 + s_label[..., 1:].argmax(-1)).tolist()
            preds = [[(i, j, labels[i][j]) for i, j in spans]
                    for spans, labels in zip(pred_spans, pred_labels)]
        else:
            bz, lens, _, lsize = s_label.shape

            # (s_label.view(bz, lens, lens, lsize, 1) + corase.view(1, 1, 1, lsize, 3)): (B, seq_len, seq_len, lsize, 3)
            # NOTE: [..., 1:, :] is used to mask first unk label, which shouldn't participate in max
            # NOTE: max(-2) and corase is used to get max labels of three sub-labels: POS*, POS, SYN/SYN*
            # pred_values, pred_labels = (B, seq_len, seq_len, 3), (B, seq_len, seq_len, 3)
            pred_values, pred_labels = (s_label.view(bz, lens, lens, lsize, 1) + corase.view(1, 1, 1, lsize, 3))[..., 1:, :].max(-2)
            # recovery order including unk
            pred_labels += 1

            def dt_func(span):
                """[summary]

                Args:
                    span (iterator): a iterator for top-down spans
                        a span is like (i, j, max_scores -> Tensor(3), label_indexes -> Tensor(3))
                Returns:
                    [type]: [description]
                """
                # (i, j, max_scores, label_indexes)
                i, j, value, label = next(span)
                # leaf node
                if j == i+1:
                    # compare POS* and POS, select larger one
                    corase_label = value[:2].argmax(-1)
                    # print(corase_label)
                    return i, j, value, label, corase_label, []
                else:
                    # How to get left child and right child's label?
                    l_i, l_j, l_v, l_l, l_c, l_t = dt_func(span)
                    r_i, r_j, r_v, r_l, r_c, r_t = dt_func(span)
                    # check children's label
                    if (l_c > 0) == (r_c > 0):
                        # all POS*
                        if l_c == 0:
                            corase_label = value[:2].argmax(-1)
                            # print(corase_label)
                        # all not POS*
                        else:
                            corase_label = 2
                    # one POS*, one not POS*(POS, SYN*, SYN)
                    else:
                        # large span
                        if j - i > self.args.alpha:
                            # label self to SYN/SYN*
                            corase_label = 2
                            # change POS* to POS
                            l_c = 1 if l_c == 0 else l_c
                            r_c = 1 if r_c == 0 else r_c
                        # small span
                        else:
                            # label self to POS? # TODO or POS*?
                            corase_label = 1
                            # change POS to POS*, # TODO SYN*, SYN? 
                            l_c = 0 if l_c == 1 else l_c
                            r_c = 0 if r_c == 1 else r_c
                    # Tensor(3) to Tensor(1), get real index of label
                    l_l = l_l[l_c]
                    r_l = r_l[r_c]
                    # print(f"{self.CHART.vocab.itos[l_l]}, {self.CHART.vocab.itos[r_l]} -> {self.CHART.vocab.itos[label[corase_label]]}")
                    # print()
                    return i, j, value, label, corase_label, [(l_i, l_j, l_l)] + l_t + [(r_i, r_j, r_l)] + r_t

            def decode_func(span, val, lab):
                """[summary]

                Args:
                    span ([type]): [description]
                    val ([type]): [description]
                    lab ([type]): [description]

                Returns:
                    [type]: [description]
                """
                # print(lab)
                i, j, _, label, corase_label, span = dt_func(iter([(i, j, val[i, j], lab[i, j]) for i, j in span]))
                return [(i, j, label[corase_label])] + span

            preds = [decode_func(spans, values, labels)
            # handle one sentence one time
                        for spans, values, labels in zip(pred_spans, pred_values, pred_labels)]

        exit()
        return preds

# def contrained_labeling(node, s):
#     """[summary]

#     Args:
#         node ([type]): current node to be label
#         s ([type]): score for labels
#     """
#     # reset label
#     node.label == None
#     # check leaf or internal node
#     if node is 'leaf':
#         if node.parent.label is None:
#             node.label = argmax(POS or POS*)
#         # if parent is not None, change node's label to POS*
#         else:
#             node.label = argmax(POS*)
#     else:
#         if node.parent.label is None:
#             # check two children's label
#             label_left = contrained_labeling(node.left_child, s)
#             label_right = contrained_labeling(node.right_child, s)
#             i, j = node.boundary
#             # two children are both POS*
#             if label_left in POS* and label_right in POS*:
#                 node.label = argmax(POS or POS*)
#             # two children are neither POS*
#             elif label_left not in POS* and label_right not in POS*:
#                 node.label = argmax(SYN or SYN*)
#             # one child is POS* and another is not POS*
#             # such as: SYN -> POS* SYN
#             else:
#                 # small span
#                 if j - i < alpha:
#                     # ... => POS/POS* -> POS* POS*
#                     node.label = argmax(POS or POS*)
#                     # change children label to POS*
#                     contrained_labeling(node.left_child, s)
#                     contrained_labeling(node.right_child, s)
#                 # large span
#                 else:
#                     # ... => SYN -> POS SYN
#                     node.label = argmax(SYN or SYN*)
#                     if label_left in POS*:
#                         node.left_child.label = argmax(POS)
#                     elif label_right in POS*:
#                         node.right_child.label = argmax(POS)

#         # if parent is not None, change node's and its children's labels to POS*
#         else:
#             node.label = argmax(POS*)
#             contrained_labeling(node.left_child, s)
#             contrained_labeling(node.right_child, s)



