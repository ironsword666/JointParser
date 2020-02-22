# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, n_labels, batch_first=True):
        super(CRF, self).__init__()

        self.n_labels = n_labels
        self.batch_first = batch_first
        self.trans = nn.Parameter(torch.Tensor(n_labels, n_labels))
        self.strans = nn.Parameter(torch.Tensor(n_labels))
        self.etrans = nn.Parameter(torch.Tensor(n_labels))

        self.reset_parameters()

    def extra_repr(self):
        s = f"n_labels={self.n_labels}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.trans)
        nn.init.zeros_(self.strans)
        nn.init.zeros_(self.etrans)

    def forward(self, emit, target, mask):
        logZ = self.get_logZ(emit, mask)
        score = self.get_score(emit, target, mask)

        return logZ - score

    def get_logZ(self, emit, mask):
        if self.batch_first:
            emit, mask = emit.transpose(0, 1), mask.t()
        seq_len, batch_size, n_labels = emit.shape

        alpha = self.strans + emit[0]  # [batch_size, n_labels]

        for i in range(1, seq_len):
            scores = self.trans + alpha.unsqueeze(-1)
            scores = torch.logsumexp(scores + emit[i].unsqueeze(1), dim=1)
            alpha[mask[i]] = scores[mask[i]]
        logZ = torch.logsumexp(alpha + self.etrans, dim=1).sum()

        return logZ / batch_size

    def get_score(self, emit, target, mask):
        if self.batch_first:
            emit, target, mask = emit.transpose(0, 1), target.t(), mask.t()
        seq_len, batch_size, n_labels = emit.shape
        scores = emit.new_zeros(seq_len, batch_size)

        # plus the transition score
        scores[1:] += self.trans[target[:-1], target[1:]]
        # plus the emit score
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # filter some scores by mask
        score = scores.masked_select(mask).sum()

        ends = mask.sum(dim=0).view(1, -1) - 1
        # plus the score of start transitions
        score += self.strans[target[0]].sum()
        # plus the score of end transitions
        score += self.etrans[target.gather(dim=0, index=ends)].sum()

        return score / batch_size
