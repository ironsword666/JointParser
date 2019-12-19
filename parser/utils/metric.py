# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import tempfile


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-5):
        super(Metric, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        return f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        arc_mask = arc_preds.eq(arc_golds)[mask]
        rel_mask = rel_preds.eq(rel_golds)[mask] & arc_mask

        self.total += len(arc_mask)
        self.correct_arcs += arc_mask.sum().item()
        self.correct_rels += rel_mask.sum().item()

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class EVALBMetric(Metric):

    def __init__(self, evalb, evalb_param, eps=1e-5):
        super(EVALBMetric, self).__init__()

        self.evalb = evalb
        self.evalb_param = evalb_param
        self.header_line = ['ID', 'Len.', 'Stat.', 'Recal',
                            'Prec.', 'Bracket', 'gold', 'test',
                            'Bracket', 'Words', 'Tags', 'Accracy']

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds, mask):
        tempdir = tempfile.mkdtemp()
        pred_path = os.path.join(tempdir, 'preds.pid')
        gold_path = os.path.join(tempdir, 'golds.pid')
        with open(pred_path, 'w') as f:
            f.writelines([f"{tree}\n" for tree in preds])
        with open(gold_path, 'w') as f:
            f.writelines([f"{tree}\n" for tree in golds])

        completed = subprocess.run([self.evalb,
                                    "-p",
                                    self.evalb_param,
                                    gold_path,
                                    pred_path],
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True,
                                   check=True)
        for line in completed.stdout.split("\n"):
            stripped = line.strip().split()
            if len(stripped) == 12 and stripped != self.header_line:
                numeric_line = [float(x) for x in stripped]
                self.tp += numeric_line[5]
                self.pred += numeric_line[7]
                self.gold += numeric_line[6]
        shutil.rmtree(tempdir)

    def __repr__(self):
        p, r, f = self.precision, self.recall, self.f_score

        return f"P: {p:6.2%} R: {r:6.2%} F: {f:6.2%}"

    @property
    def score(self):
        return self.f_score

    @property
    def precision(self):
        return self.tp / (self.pred + self.eps)

    @property
    def recall(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f_score(self):
        precision = self.precision
        recall = self.recall

        return 2 * precision * recall / (precision + recall + self.eps)
