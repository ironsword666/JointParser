# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable
from parser.utils.field import Field

from nltk.tree import Tree

Treebank = namedtuple(typename='Treebank',
                      field_names=['WORD', 'POS', 'TREE'],
                      defaults=[None]*3)


class Sentence(object):

    def __init__(self, tree, fields):
        self.tree = tree
        self.fields = [field if isinstance(field, Iterable) else [field]
                       for field in fields]
        self.values = [*zip(*tree.pos()), tree]
        for field, value in zip(self.fields, self.values):
            for f in field:
                setattr(self, f.name, value)

    def __len__(self):
        return len(list(self.tree.leaves()))

    def __repr__(self):
        return self.tree.pformat(1000000)


class Corpus(object):

    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)

    def __setattr__(self, name, value):
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod
    def load(cls, path, fields):
        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]
        with open(path, 'r') as f:
            trees = [Tree.fromstring(string) for string in f]
        sentences = [Sentence(tree, fields) for tree in trees]

        return cls(fields, sentences)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")
