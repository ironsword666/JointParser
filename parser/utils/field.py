# -*- coding: utf-8 -*-

from collections import Counter
from itertools import product

from numpy import dtype

from parser.utils.common import pos_label
from parser.utils.fn import tohalfwidth, binarize, add_child
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

        # self.coarse_labels = ['POS*', 'POS', 'SYN*', 'SYN', 'UnaryPOS', 'UnarySYN']
        self.coarse_labels = ['POS*', 'POS', 'SYN*', 'SYN']

        # mask tensor
        # self.coarse_mask, self.unary_mask = self.get_coarse_mask(corpus)


    def get_coarse_mask(self, coarse_productions, corpus=None):

        label_dict = {k : v for v, k in enumerate(self.coarse_labels)}

        n = len(self.coarse_labels)

        # (n_coarse, n_coarse, n_coarse)
        coarse_mask = torch.full((n, n, n),  float('-inf'), dtype=torch.float)
        # (n_coarse)
        unary_mask = torch.full((n,), float('-inf'), dtype=torch.float)

        # coarse_productions = self.get_coarse_productions(corpus)

        for p in coarse_productions:
            # str to index
            k = label_dict[p[0]]
            i = label_dict[p[1]]
            j = label_dict[p[2]]
            coarse_mask[i, j, k] = 0

        # for l in ['POS*', 'POS', 'UnaryPOS']:
        for l in ['POS*', 'POS']:

            unary_mask[label_dict[l]] = 0

        # print(coarse_mask)
        # print(unary_mask)
        
        return coarse_mask, unary_mask

    def get_coarse_productions(self, corpus):

        coarse_productions = set()
        # unary_productions = set()
        # char-tree un-cnf
        for tree in corpus.trees:
            # cnf
            tree = binarize(tree)[0]
            productions = tree.productions()
            # Production
            for p in productions:
                # Symbol() to str
                left = p.lhs().symbol()
                # ignore `CHAR`
                if left == 'CHAR':
                    continue
                # list[str]
                right = [s.symbol() if not isinstance(s, str) else s for s in p.rhs()]
                # A->word
                if len(right) == 1 and right[0] == 'CHAR':
                    # if self.fine2coarse_label(left) == 'SYN' or self.fine2coarse_label(left) == 'SYN*':
                    #     print(p)
                    # unary_productions.add(self.fine2coarse_label(left))
                    continue
                # coarse_production
                p = self.fine2coarse_production(left, right)

                # if p[0] == 'SYN' and p[1] == 'POS' and p[2] == 'SYN':
                    # print(left, right)
                    # tree.pretty_print()
                coarse_productions.add(p)

        coarse_productions = list(coarse_productions)
        coarse_productions.sort(key = lambda x: (x[0], x[1], x[2]))
        for p in coarse_productions:
            print(p)

        return coarse_productions

    def fine2coarse_production(self, left, right):
        """
        Args:
            left (str): left side of a production 
            right (list[str]): right side of a production 

        Returns:
            (left, child_one, child_two)
        """


        left = self.fine2coarse_label(left)

        right = [self.fine2coarse_label(l) for l in right]

        return (left, *right)

    def fine2coarse_label(self, label):
        """
        Args:
            label (str): fine-grained label
        
        Return:
            (str): coarse-grained label
        """

        idx =  self.sublabel_cluster(label)

        return self.coarse_labels[idx]


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

    def sublabel_cluster(self, label=None):
        """cluster full label to four sub label.

        Args:
            label (str): full_label

        Returns:
            [int]: 1,2,3,4 for POS*, POS, SYN*, SYN
        """
        if label is None:
            return 4
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
            # POS
            if label[-1] in pos_label:
                return 1
            # SYN
            else:
                return 3
            # if len(label) == 1:
            #     # POS
            #     if label[-1] in pos_label:
            #         return 1
            #     # SYN
            #     else:
            #         return 3
            # # check unary rule
            # else:
            #     # SYNs+POS
            #     if label[-1] in pos_label:
            #         return 4
            #     # SYNs+SYN
            #     else:
            #         return 5

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
            label_chart = torch.full((seq_len, seq_len), -1, dtype=torch.long)
            # sequence = add_child(iter(sequence))[1]
            # for i, j, (left, right, label) in sequence:
            for i, j, label in sequence:

                # idx_l, idx_r, idx = self.sublabel_cluster(left), self.sublabel_cluster(right), self.sublabel_cluster(label)
                # span_chart[i, j] = torch.tensor([idx_l, idx_r, idx])
                span_chart[i, j] = self.sublabel_cluster(label)
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
