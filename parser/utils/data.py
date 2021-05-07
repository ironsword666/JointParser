# -*- coding: utf-8 -*-

from collections.abc import Iterable
from itertools import chain
from parser.utils.alg import kmeans
from parser.utils.field import Field
from parser.utils.fn import pad

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class TextDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(TextDataLoader, self).__init__(*args, **kwargs)

        self.fields = self.dataset.fields

    def __iter__(self):
        # raw_batch is result of collect_fn
        for raw_batch in super(TextDataLoader, self).__iter__():
            batch, device = [], 'cuda' if torch.cuda.is_available() else 'cpu'
            # pad list of tensor to tensor
            for data, field in zip(raw_batch, self.fields):
                if isinstance(field, Field):
                    if isinstance(data[0], torch.Tensor):
                        data = pad(data, field.pad_index).to(device)
                    # such as chart: [(span, label), ]
                    elif isinstance(data[0], Iterable):
                        data = [pad(f, field.pad_index).to(device)
                                for f in zip(*data)]
                batch.append(data)
            yield batch


class TextDataset(Dataset):

    def __init__(self, corpus, fields, n_buckets=1):
        super(TextDataset, self).__init__()

        self.corpus = corpus
        # list[Field()]
        self.fields = list(chain(*[
            field if isinstance(field, Iterable) else [field]
            for field in fields if field is not None
        ]))
        # NOTE: dataset have numericalized data, list of tensor
        for field in self.fields:
            setattr(self,
                    field.name,
                    field.transform(getattr(corpus, field.name)))
        # NOTE: the final bucket count is roughly equal to n_buckets
        # list[int]: record sentence length 
        self.lengths = [len(i) + sum([bool(field.bos), bool(field.bos)])
                        for i in corpus]
        # put sentences with near length to a cluster, a bucket is considered as a cluster
        # dict{center_length: iterable(index)}, index corresponds to  a sentence in the corpus
        self.buckets = dict(zip(*kmeans(self.lengths, n_buckets)))

    def __getitem__(self, index):
        for field in self.fields:
            yield getattr(self, field.name)[index]

    def __len__(self):
        return len(self.corpus)

    @property
    def loader(self):
        if hasattr(self, 'data_loader'):
            return self.data_loader
        else:
            raise AttributeError

    @loader.setter
    def loader(self, data_loader):
        self.data_loader = data_loader

    @classmethod
    def collate_fn(cls, batch):
        # batch = dataset[batch], is list[(tree, char, pos, chart)]

        # (tree, ), (char, ), (pos, ), (chart, )
        return (field for field in zip(*batch))


class TextSampler(Sampler):

    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # iterable like: [center_length],  [cluster: [sent_length], ]
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        # the number of chunks in each bucket, which is clipped by
        # range [1, len(bucket)]
        # a cluster can be split into xx chunks, each chunk contain about 5000 words
        # len(chunks) == len(buckets)
        self.chunks = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]

    def __iter__(self):
        # if shuffle, shuffle both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        # i is index of cluster, shuffle or not shuffle
        for i in range_fn(len(self.buckets)).tolist():
            # chunk length of each chunk of a cluster
            # [chunk_length]
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                # indexes of a batch of sentences
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        # how many batches of a corpus
        return sum(self.chunks)


def batchify(dataset, batch_size, shuffle=False, num_workers=0):
    batch_sampler = TextSampler(buckets=dataset.buckets,
                                batch_size=batch_size,
                                shuffle=shuffle)
    loader = TextDataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=dataset.collate_fn,
                            num_workers=num_workers)

    return loader
