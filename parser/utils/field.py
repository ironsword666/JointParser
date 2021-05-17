# -*- coding: utf-8 -*-

from collections import Counter

from numpy import dtype
from parser.utils.common import pos_label
from parser.utils.fn import tohalfwidth
from parser.utils.vocab import Vocab

import torch


class RawField(object):

    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        return [self.preprocess(sequence) for sequence in sequences]


class Field(RawField):

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, tohalfwidth=False, use_vocab=True, tokenize=None, fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.tohalfwidth = tohalfwidth
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.tohalfwidth:
            params.append(f"tohalfwidth={self.tohalfwidth}")
        s += f", ".join(params)
        s += f")"

        return s

    @property
    def pad_index(self):
        return self.specials.index(self.pad) if self.pad is not None else 0

    @property
    def unk_index(self):
        return self.specials.index(self.unk) if self.unk is not None else 0

    @property
    def bos_index(self):
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        return self.specials.index(self.eos)

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]
        if self.tohalfwidth:
            sequence = [tohalfwidth(token) for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter(token
                          for sequence in sequences
                          for token in self.preprocess(sequence))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class NGramField(Field):
    def __init__(self, *args, **kwargs):
        self.n = kwargs.pop('n') if 'n' in kwargs else 1
        super(NGramField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1, dict_file=None, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter()
        sequences = [self.preprocess(sequence) for sequence in sequences]
        n_pad = (self.n - 1)
        for sequence in sequences:
            chars = sequence + [self.eos] * n_pad
            bichars = ["".join(chars[i + s] for s in range(self.n))
                       for i in range(len(chars) - n_pad)]
            counter.update(bichars)
        if dict_file is not None:
            counter &= self.read_dict(dict_file)
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)
        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def read_dict(self, dict_file):
        word_list = dict()
        with open(dict_file, encoding='utf-8') as dict_in:
            for line in dict_in:
                line = line.split()
                if len(line) == 3:
                    word_list[line[0]] = 100
        return Counter(word_list)

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        params.append(f"n={self.n}")
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.tohalfwidth:
            params.append(f"tohalfwidth={self.tohalfwidth}")
        s += f", ".join(params)
        s += f")"

        return s

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        n_pad = (self.n - 1)
        for sent_idx, sequence in enumerate(sequences):
            chars = sequence + [self.eos] * n_pad
            sequences[sent_idx] = ["".join(chars[i + s] for s in range(self.n))
                                   for i in range(len(chars) - n_pad)]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class ChartField(Field):

    def build(self, corpus, min_freq=1):
        sequences = getattr(corpus, self.name)
        counter = Counter(label
                          for sequence in sequences
                          for i, j, label in self.preprocess(sequence))
        meta_labels = Counter({label.split(
            "+")[-1]: min_freq for label, freq in counter.items() if freq < min_freq})
        counter |= meta_labels
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index, keep_sorted_label=True)

    def label_cluster(self, label):
        # fake label 
        if label.endswith("|<>"):
            label = label[:-3].split("+")[-1]
            # POS*
            if label in pos_label:
                return 0
            # SYN*
            else:
                return 2
        else:
            label = label.split("+")[-1]
            # POS
            if label in pos_label:
                return 1
            # SYN
            else:
                return 2

    def get_label_index(self, label):
        if label in self.vocab:
            return self.vocab[label]
        else:
            label_set = set(label.split("+")[:-1])
            last_state = label.split("+")[-1]
            for l_set, whole_l, last_l in self.vocab.sorted_label:
                if last_state == last_l and len(l_set - label_set) <= 0:
                    return self.vocab[whole_l]
            return self.vocab[last_state]

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans, labels = [], []

        for sequence in sequences:
            seq_len = sequence[0][1] + 1
            span_chart = torch.full((seq_len, seq_len), self.pad_index).bool()
            label_chart = torch.full((seq_len, seq_len), self.pad_index).long()
            for i, j, label in sequence:
                span_chart[i, j] = 1
                label_chart[i, j] = self.get_label_index(label)
            spans.append(span_chart)
            labels.append(label_chart)
        return list(zip(spans, labels))

class SubLabelField(ChartField):
    """
    Processing (i, j, label, sub_label)

    """

    def build(self, corpus, min_freq=1):
        sequences = getattr(corpus, self.name)
        counter = Counter(label
                          for sequence in sequences
                          for i, j, label in self.preprocess(sequence))
        meta_labels = Counter({label.split(
            "+")[-1]: min_freq for label, freq in counter.items() if freq < min_freq})
        counter |= meta_labels
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index, keep_sorted_label=True)

    def statistic(self):
        sub_pos, pos, sub_syn, syn, unary_pos, unary_syn = [], [], [], [], [], []
        print(len(self.vocab))
        for label in self.vocab.itos:
            idx = self.sublabel_cluster(label)
            if idx == 0:
                sub_pos.append(label)
            elif idx == 1:
                pos.append(label)
            elif idx == 2:
                sub_syn.append(label)
            elif idx == 3:
                syn.append(label)
            elif idx == 4:
                unary_pos.append(label)
            elif idx == 5:
                unary_syn.append(label)

        print('------------')
        print('POS*:')
        print(len(sub_pos))
        for l in sub_pos:
            print(l)
        print('------------')
        print('POS:')
        print(len(pos))
        for l in pos:
            print(l)
        print('------------')
        print('SYN*:')
        print(len(sub_syn))
        for l in sub_syn:
            print(l)
        print('------------')
        print('SYN:')
        print(len(syn))
        for l in syn:
            print(l)
        print('------------')
        print('Unary_POS:')
        print(len(unary_pos))
        for l in unary_pos:
            print(l)
        print('------------')
        print('Unary_SYN:')
        print(len(unary_syn))
        for l in unary_syn:
            print(l)
        # exit()

    def label_cluster(self, label):
        # fake label 
        if label.endswith("|<>"):
            label = label[:-3].split("+")[-1]
            # POS*
            if label in pos_label:
                return 0
            # SYN*
            else:
                return 2
        else:
            label = label.split("+")[-1]
            # POS
            if label in pos_label:
                return 1
            # SYN
            else:
                return 2

    def sublabel_cluster(self, label):
        """cluster full label to four sub label.

        Args:
            label (str): full_label

        Returns:
            [int]: 1,2,3,4 for POS*, POS, SYN*, SYN
        """
        # dummy label
        if label.endswith("|<>"):
            label = label[:-3].split("+")[-1]
            # POS*
            if label in pos_label:
                return 0
            # SYN*
            else:
                return 2
        else:
            
            label = label.split("+")
            if len(label) == 1:
                # POS
                if label[-1] in pos_label:
                    return 1
                # SYN
                else:
                    return 3
            # check unary rule
            else:
                # SYNs+POS
                if label[-1] in pos_label:
                    return 4
                # SYNs+SYN
                else:
                    return 5

    def get_label_index(self, label):
        """
        Get label index, if doesn't exist,
        project low frequent label to high frequent label.
        """
        if label in self.vocab:
            return self.vocab[label]
        else:
            label_set = set(label.split("+")[:-1])
            last_state = label.split("+")[-1]
            for l_set, whole_l, last_l in self.vocab.sorted_label:
                if last_state == last_l and len(l_set - label_set) <= 0:
                    return self.vocab[whole_l]
            return self.vocab[last_state]

    def get_sublabel_index(self, label):

        # high frequent label which is in vocab
        if label in self.vocab:
            return self.sublabel_cluster(label)
        # project low frequent label to high frequent label
        else:
            # split POS and SYNs
            label_set = set(label.split("+")[:-1])
            last_state = label.split("+")[-1]
            # compare POS and SYNs with high frequent label
            for l_set, whole_l, last_l in self.vocab.sorted_label:
                if last_state == last_l and len(l_set - label_set) <= 0:
                    return self.sublabel_cluster(whole_l)
            return self.sublabel_cluster(last_state)

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans, labels = [], []

        for sequence in sequences:
            seq_len = sequence[0][1] + 1
            # 0,1,2,3,4 only need 8-bit and 0 indicates span(i, j) is not a constituent
            span_chart = torch.full((seq_len, seq_len), -1, dtype=torch.long)
            # pad 0 indicates span(i, j) is not a constituent
            label_chart = torch.full((seq_len, seq_len), self.pad_index, dtype=torch.long)
            for i, j, label in sequence:
                span_chart[i, j] = self.get_sublabel_index(label)
                label_chart[i, j] = self.get_label_index(label)
            spans.append(span_chart)
            labels.append(label_chart)
        return list(zip(spans, labels))


class BertField(Field):

    def transform(self, sequences):
        subwords, lens = [], []
        sequences = [list(sequence)
                     for sequence in sequences]

        for sequence in sequences:
            sequence = self.preprocess(sequence)
            sequence = [piece if piece else self.preprocess(self.pad)
                        for piece in sequence]
            subwords.append(sequence)
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).gt(0) for pieces in subwords]

        return list(zip(subwords, mask))
