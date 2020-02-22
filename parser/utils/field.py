# -*- coding: utf-8 -*-

from collections import Counter
from parser.utils.vocab import Vocab
from parser.utils.fn import tohalfwidth
from parser.utils.common import bos, eos
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
        n_pad = self.n - 1
        for sequence in sequences:
            chars = list(sequence) + [eos] * n_pad
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
            chars = list(sequence) + [eos] * n_pad
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


class BertField(Field):

    def transform(self, sequences):
        subwords, lens = [], []
        sequences = [([self.bos] if self.bos else []) + list(sequence) +
                     ([self.eos] if self.eos else [])
                     for sequence in sequences]

        for sequence in sequences:
            sequence = [self.preprocess(token) for token in sequence]
            sequence = [piece if piece else self.preprocess(self.pad)
                        for piece in sequence]
            subwords.append(sum(sequence, []))
            lens.append(torch.tensor([len(piece) for piece in sequence]))
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).gt(0) for pieces in subwords]

        return list(zip(subwords, lens, mask))
