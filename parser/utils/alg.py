# -*- coding: utf-8 -*-

from ctypes.wintypes import tagRECT
from parser.utils.fn import stripe

import torch
import torch.autograd as autograd


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


@torch.enable_grad()
def crf(scores, mask, target=None, marg=False):
    """[summary]

    Args:
        scores (Tensor(B, seq_len, seq_len, n_labels))
        mask (Tensor(B, seq_len, seq_len))
        target (Tensor(B, seq_len, seq_len).int8): Defaults to None.
        marg (bool, optional): Defaults to False.

    Returns:
        crf-loss, marginal probability for spans
    """
    # (B)
    lens = mask[:, 0].sum(-1)
    total = lens.sum()
    batch_size, seq_len, seq_len, n_labels = scores.shape
    # in eval(), it's false; and in train(), it's true
    training = scores.requires_grad
    # always enable the gradient computation of scores
    # in order for the computation of marginal probs.
    # requires_grad_(requires_grad=True):
    # Change if autograd should record operations on scores: 
    #   sets scores’s requires_grad attribute in-place. Returns this tensor.
    # (seq_len, seq_len, n_labels,  B)
    s = inside(scores.requires_grad_(), mask)
    # expand lens to Tensor(1, n_labels, B)
    lens = lens.view(1, 1, batch_size).expand(-1, n_labels, -1)
    # get alpha(0, length, l) for each sentence
    # (seq_len, n_labels, B).gather(0, Tensor(1, n_labels, B)) -> Tensor(1, n_labels, B)
    logZ = s[0].gather(0, lens).logsumexp(1).sum()
    # marginal probs are used for decoding, and can be computed by
    # combining the inside algorithm and autograd mechanism
    # instead of the entire inside-outside process.
    probs = scores
    if marg:
        # Computes and returns the sum of gradients of outputs w.r.t. the inputs.
        # retain_graph: If False, the graph used to compute the grad will be freed.
        # Tensor(B, seq_len, seq_len, n_labels)
        probs, = autograd.grad(logZ, scores, retain_graph=training)
    if target is None:
        return probs
    # (B, seq_len, seq_len)
    span_mask = (target > 0) & mask
    # (T, n_labels)
    scores = scores[span_mask] 
    # (T, 1)
    target = target[span_mask].unsqueeze(-1) - 1
    # TODO why / total?
    loss = (logZ - scores.gather(1, target).sum()) / total
    return loss, probs


def inside(scores, mask):
    """Inside algorithm.

    Args:
        scores (Tensor(B, seq_len, seq_len, n_labels))
        mask (Tensor(B, seq_len, seq_len))

    Returns:
        Tensor: [seq_len, seq_len, n_labels, batch_size]
    """
    batch_size, seq_len, _ = scores.shape
    # permute is convenient for diagonal which acts on dim1=0 and dim2=1
    # [seq_len, seq_len, n_labels, batch_size]
    scores, mask = scores.permute(1, 2, 3, 0), mask.permute(1, 2, 0)
    # s[i, j, l]: rooted in label `l` and span from i to j
    # [seq_len, seq_len, n_labels, batch_size]
    s = torch.full_like(scores, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # diag_mask is used for ignoring the excess of each sentence
        # [batch_size, n]
        diag_mask = mask.diagonal(w)

        if w == 1:
            # scores.diagonal(w): [batch_size, n_labels, n]
            # scores.diagonal(w).permute(1, 2, 0)[diag_mask]: (T, n_labels)
            s.diagonal(w).permute(1, 2, 0)[diag_mask] = scores.diagonal(w).permute(1, 2, 0)[diag_mask]
            continue 

        # scores for sub-tree spanning from i to k, considering all labels (by logsumexp)
        # NOTE: stripe considering all split points and spans with same width
        # stripe: [n, w-1, n_labels, batch_size] -> logsumexp: [n, w-1, batch_size] 
        s_left = stripe(s, n, w-1, (0, 1)).logsumexp(-2)
        s_right = stripe(s, n, w-1, (1, w), 0).logsumexp(-2)
        # left sub-tree and right sub-tree
        # [n, w-1, batch_size] 
        s_span = s_left + s_right
        # [batch_size, n, w-1]
        s_span = s_span.permute(2, 0, 1)
        # (T, w-1) -> (T)
        s_span = s_span[diag_mask].logsumexp(-1)
        # (T_labels) = (T, 1) + (T, n_labels)
        s.diagonal(w).permute(1, 2, 0)[diag_mask] = s_span.unsqueeze(-1) + scores.diagonal(w).permute(1, 2, 0)[diag_mask]

    # [seq_len, seq_len, n_labels, batch_size]
    return s


def cky(scores, mask):
    lens = mask[:, 0].sum(-1)
    scores = scores.permute(1, 2, 0)
    seq_len, seq_len, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p.new_tensor(range(n)).unsqueeze(0)

        if w == 1:
            s.diagonal(w).copy_(scores.diagonal(w))
            continue
        # [n, w, batch_size]
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(w).copy_(s_span + scores.diagonal(w))
        p.diagonal(w).copy_(p_span + starts + 1)

    def backtrack(p, i, j):
        if j == i + 1:
            return [(i, j)]
        split = p[i][j]
        ltree = backtrack(p, i, split)
        rtree = backtrack(p, split, j)
        # top-down, [(0, 9), (0, 6), (0, 3), ]
        return [(i, j)] + ltree + rtree

    p = p.permute(2, 0, 1).tolist()
    trees = [backtrack(p[i], 0, length)
             for i, length in enumerate(lens.tolist())]

    return trees
