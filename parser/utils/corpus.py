# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable
from itertools import chain
from parser.utils.tree import load_trees

Treebank = namedtuple(typename='Treebank',
                      field_names=['WORD', 'POS', 'LABEL'],
                      defaults=[None]*3)


class Sentence(object):

    def __init__(self, tree, fields):
        self.tree = tree.convert()
        self.fields = [field if isinstance(field, Iterable) else [field]
                       for field in fields]

        self.values = [[leaf.word for leaf in self.tree.leaves()],
                       [leaf.tag for leaf in self.tree.leaves()],
                       [self.tree.oracle_label(i, j)
                        for i in range(0, len(self))
                        for j in range(i+1, len(self)+1)]]
        for field, value in zip(fields, self.values):
            if isinstance(field, Iterable):
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)

    def __len__(self):
        return len(list(self.tree.leaves()))

    def __repr__(self):
        return self.tree.convert()


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
        trees = load_trees(path)
        sentences = [Sentence(tree, fields) for tree in trees]

        return cls(fields, sentences)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")
