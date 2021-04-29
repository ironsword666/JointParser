# -*- coding: utf-8 -*-

import os
from datetime import datetime
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify

import torch


class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--fdata', default='data/ctb51/test.pid',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.pid',
                               help='path to predicted result')

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus,
                              self.fields[:-1],
                              args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Make predictions on the dataset")
        start = datetime.now()
        pred_trees = self.predict(dataset.loader)
        total_time = datetime.now() - start
        # restore the order of sentences in the buckets
        indices = torch.tensor([i
                                for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()
        corpus.trees = [pred_trees[i] for i in indices]
        print(f"Save the predicted result to {args.fpred}")
        # create dir
        path_list = (args.fpred).split(r'/')
        path_list.pop()
        path = r'/'.join(path_list)
        if not os.path.exists(path):
            os.mkdir(path)
            
        corpus.save(args.fpred)
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
