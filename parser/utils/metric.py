# -*- coding: utf-8 -*-
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


class AttachmentMethod(Metric):

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


class F1Method(Metric):

    def __init__(self, ignore_index=0, eps=1e-5):
        super(F1Method, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.ignore_index = ignore_index
        self.eps = eps

    def __call__(self, preds, golds, mask):
        pred_mask = preds.ne(self.ignore_index) & mask
        gold_mask = golds.ne(self.ignore_index) & mask
        self.tp += (preds.eq(golds) & pred_mask).sum().item()
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()

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
