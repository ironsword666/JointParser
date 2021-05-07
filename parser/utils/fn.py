# -*- coding: utf-8 -*-

import unicodedata

from nltk.tree import Tree
from parser.utils.common import pos_label


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

    # TODO usage and how work

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


def binarize(tree):
    """transform a CFG tree to a CNF tree
    
    if a tree has two more children and a child's child is a terminal:
        add a artificial parent node A* to this child.

    # TODO
    the operator is to 

                 |
                IP
        _________|__________________
        |              VP            |
        |          ____|____         |
        NP       ADVP       VP       |
        |         |         |        |
        NN        AD        VA       PU
    ____|____     |     ____|___     |
    CHAR CHAR CHAR CHAR CHAR     CHAR CHAR
    |    |    |    |    |        |    |
    巧    克    力    很    美        味    .

                              |                           
                             IP                         
                        ______|_______________________    
                    IP|<>                            |  
                _____|_____________                 |   
                NP+NN                 VP               |  
            _____|_____        ______|_____           |   
        NN|<>         |      |          VP+VA        |  
    _____|_____      |      |       _____|_____     |   
    NN|<>       NN|<> NN|<> ADVP+AD VA|<>       VA|<>  PU 
    |           |     |      |      |           |    |   
    CHAR        CHAR  CHAR   CHAR   CHAR        CHAR CHAR
    |           |     |      |      |           |    |   
    巧           克     力      很      美           味    .  

    Args:
        tree ([type]): [description]

    Returns:
        Tree: a 
    """
    tree = tree.copy(True)
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            if len(node) > 1:
                for i, child in enumerate(node):
                    if not isinstance(child[0], Tree):
                        node[i] = Tree(f"{node.label()}|<>", [child])
    tree.chomsky_normal_form('left', 0, 0)
    tree.collapse_unary()

    return tree


def decompose(tree):
    """transform a word tree to a char tree

    For example:
    `((IP (NP (NN 巧克力)) (VP (ADVP (AD 很)) (VP (VA 美味))) (PU .)))`

                |
                IP
        _________|__________________
        |              VP            |
        |          ____|____         |
        NP       ADVP       VP       |
        |         |         |        |
        NN        AD        VA       PU
    ____|____     |     ____|___     |
    CHAR CHAR CHAR CHAR CHAR     CHAR CHAR
    |    |    |    |    |        |    |
    巧    克    力    很    美        味    .

    Args:
        tree (Tree): word tree 
                                                    
    Returns:
        Tree: char tree, pre-terminal of each char is `CHAR`
        List: POS list of word tree (not char tree)
    """
    tree = tree.copy(True)
    pos = set(list(zip(*tree.pos()))[1])
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            # add all subtree to stack
            nodes.extend([child for child in node])
            for i, child in enumerate(node):
                # check subtree: POS->WORD 
                if isinstance(child, Tree) and len(child) == 1 and isinstance(child[0], str):
                    # replace node[i] with a new created Tree()
                    # so node[i] is still in nodes and will be checked
                    # but node[i] won't affect tree anymore
                    node[i] = Tree(child.label(), [Tree("CHAR", [char])
                                                   for char in child[0]])

    return tree, pos


def compose(tree):
    tree = tree.copy(True)
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            for i, child in enumerate(node):
                if isinstance(child, Tree) and all([isinstance(grand[0], str) for grand in child]):
                    node[i] = Tree(child.label(), ["".join(child.leaves())])

    return tree


def factorize(tree, delete_labels=None, equal_labels=None):
    """Get all spans of a CNF tree but ignore the span only have one terminal (which is usually POS constituent)

    For word-level parsing, ignore POS spans,
    but for char-level parsing, `CHAR` will be ignored but POS* spans will be saved.

    For example:
    [(0, 7, 'IP'), (0, 6, 'IP|<>'), (0, 3, 'NP+NN'), (0, 2, 'NN|<>'), 
    (0, 1, 'NN|<>'), (1, 2, 'NN|<>'), (2, 3, 'NN|<>'), (3, 6, 'VP'), 
    (3, 4, 'ADVP+AD'), (4, 6, 'VP+VA'), (4, 5, 'VA|<>'), (5, 6, 'VA|<>'), (6, 7, 'PU')]

    We can get sentence length by check list[0]

    Args:
        tree ([type]): [description]
        # TODO what delete_labels do?
        delete_labels ([type], optional): [description]. Defaults to None.
        equal_labels ([type], optional): [description]. Defaults to None.
    """
    def track(tree, i):
        label = tree.label()
        if delete_labels is not None and label in delete_labels:
            label = None
        if equal_labels is not None:
            label = equal_labels.get(label, label)
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i+1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [(i, j, label)] + spans
        return j, spans
    return track(tree, 0)[1]


def build(tree, sequence):
    label = tree.label()
    leaves = [subtree for subtree in tree.subtrees()
              if not isinstance(subtree[0], Tree)]

    def recover(label, children):
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