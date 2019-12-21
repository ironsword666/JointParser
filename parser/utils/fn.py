# -*- coding: utf-8 -*-

import unicodedata

from nltk.tree import Tree


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A']
               for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char)
               for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char)
               for char in token)


def tohalfwidth(token):
    return unicodedata.normalize('NFKC', token)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.

    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def pad(tensors, padding_value=0):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def factorize(tree, i):
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        return []
    j, spans = i, []
    for child in tree:
        s = factorize(child, j)
        j = s[0][1] if s else j+1
        spans += s
    return [(i, j, tree.label())] + spans


def build(tree, sequence, nul):
    label = tree.label()
    leaves = [subtree for subtree in tree.subtrees()
              if not isinstance(subtree[0], Tree)]

    def recover(label, children):
        if label == nul:
            return children
        sublabels = [l for l in label.split('+') if not l.endswith('|<>')]
        if not sublabels:
            return children
        tree = Tree(sublabels[-1], children)
        for sublabel in reversed(sublabels[:-1]):
            tree = Tree(sublabel, [tree])
        return [tree]

    def track(node):
        i, j, label = next(node)
        if j == i+1:
            return recover(label, [leaves[i]])
        else:
            return recover(label, track(node) + track(node))

    tree = Tree(label, track(iter(sequence)))

    return tree


def satisfy_assumption(tree, suffix="|<>"):
    nodeList = [(tree, [tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):
            originalNode = node.label()
            for child in node:
                nodeList.append((child, parent))
            if len(node) > 1:
                numChildren = len(node)
                for i in range(numChildren):
                    if not isinstance(node[i][0], Tree):
                        newHead = f"{originalNode}{suffix}"
                        newNode = Tree(newHead, [node[i]])
                        node[i] = newNode
