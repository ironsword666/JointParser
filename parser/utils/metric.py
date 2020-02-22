# -*- coding: utf-8 -*-

from collections import Counter
from parser.utils.common import pos_label


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

    def __init__(self, eps=1e-8):
        super(AttachmentMetric, self).__init__()

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


class BracketMetric(Metric):

    def __init__(self, pos_label, eps=1e-8):
        super(BracketMetric, self).__init__()

        self.n = 0.0
        self.ltp = 0.0
        self.mtp = 0.0
        self.cltp = 0.0
        self.pltp = 0.0
        self.sltp = 0.0
        self.pred = 0.0
        self.cpred = 0.0
        self.spred = 0.0
        self.gold = 0.0
        self.cgold = 0.0
        self.sgold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            apred = Counter(pred)
            agold = Counter(gold)

            cpred = Counter([(i, j, label)
                             for i, j, label in pred if label not in pos_label])
            cgold = Counter([(i, j, label)
                             for i, j, label in gold if label not in pos_label])

            ucpred = Counter([(i, j)
                              for i, j, label in pred if label not in pos_label])
            uppred = Counter([(i, j)
                              for i, j, label in pred if label in pos_label])
            ucgold = Counter([(i, j)
                              for i, j, label in gold if label not in pos_label])
            upgold = Counter([(i, j)
                              for i, j, label in gold if label in pos_label])

            ppred = set(pred)
            pgold = set([(i, j, label)
                         for i, j, label in gold if label in pos_label])

            spred = [0] + sorted(set([j for _, j, _ in pred]))
            spred = set((spred[i], spred[i+1]) for i in range(len(spred) - 1))

            sgold = [0] + sorted(set([j for _, j, _ in gold]))
            sgold = set((sgold[i], sgold[i+1]) for i in range(len(sgold) - 1))

            ltp = list((apred & agold).elements())
            mtp = list((((uppred - upgold) & ucgold) |
                        ((ucpred - ucgold) & upgold)).elements())
            cltp = list((cpred & cgold).elements())
            pltp = list((ppred & pgold))
            sltp = list((spred & sgold))
            self.n += 1
            self.ltp += len(ltp)
            self.mtp += len(mtp)
            self.cltp += len(cltp)
            self.pltp += len(pltp)
            self.sltp += len(sltp)

            pred_span_count = len(pred)
            gold_span_count = len(gold)
            pred_word_count = len(spred)
            gold_word_count = len(sgold)

            self.pred += pred_span_count
            self.gold += gold_span_count
            self.cpred += pred_span_count - pred_word_count
            self.cgold += gold_span_count - gold_word_count
            self.spred += pred_word_count
            self.sgold += gold_word_count

    def __repr__(self):
        s = f"CLP: {self.clp:6.2%} CLR: {self.clr:6.2%} CLF: {self.clf:6.2%} "
        s += f"POS P: {self.plp:6.2%} POS R: {self.plr:6.2%} POS F: {self.plf:6.2%} "
        s += f"SEG P: {self.slp:6.2%} SEG R: {self.slr:6.2%} SEG F: {self.slf:6.2%} "
        s += f"MISPLACE: {self.misplace:6.2%} OVERALL F: {self.score:6.2%} "
        return s

    @property
    def score(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)

    @property
    def misplace(self):
        return self.mtp / (self.gold + self.eps)

    @property
    def clp(self):
        return self.cltp / (self.cpred + self.eps)

    @property
    def clr(self):
        return self.cltp / (self.cgold + self.eps)

    @property
    def clf(self):
        return 2 * self.cltp / (self.cpred + self.cgold + self.eps)

    @property
    def plp(self):
        return self.pltp / (self.spred + self.eps)

    @property
    def plr(self):
        return self.pltp / (self.sgold + self.eps)

    @property
    def plf(self):
        return 2 * self.pltp / (self.spred + self.sgold + self.eps)

    @property
    def slp(self):
        return self.sltp / (self.spred + self.eps)

    @property
    def slr(self):
        return self.sltp / (self.sgold + self.eps)

    @property
    def slf(self):
        return 2 * self.sltp / (self.spred + self.sgold + self.eps)
