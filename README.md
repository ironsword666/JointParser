# JointPaser

Source code for CoNLL-2021's paper [A Coarse-to-Fine Labeling Framework for Joint Word Segmentation, POS Tagging, and Constituent Parsing]().

The code of the joint framework is available at [`joint-parser`](https://github.com/ironsword666/JointParser/tree/joint-parser) branch, and the code of the pipeline framework is available at [`pipeline-cws`](https://github.com/ironsword666/JointParser/tree/pipeline-cws), [`pipeline-pos`](https://github.com/ironsword666/JointParser/tree/pipeline-pos), [`pipeline-parser`](https://github.com/ironsword666/JointParser/tree/pipeline-parser) for **Chinese Word Segmentation**, **POS Tagging**, and **Constituent Parsing** respectively.

## Requirements

As a prerequisite, the following requirements should be satisfied:

- **python >= 3.6** to support `f'str'` 

- **NLTK == 3.5** to build constituent trees

- **transformers == 3.1.0** to use BERT, and the code is not compatible with **transformers == 4.0**

- **pytorch >= 1.7.0** to use `amax()` function

## Train

**example**

```sh
python -u run.py train \
    --conf=config.ini \
    --preprocess \
    -device=0 \
    --marg \
    --mask_inside \
    --feat=bigram \
    --ftrain=/data/ctb51/train.pid \
    --fdev=/data/ctb51/dev.pid \
    --ftest=/data/ctb51/test.pid \
    --file=exp/ctb51.joint.bigram \
    > log/ctb51.joint.bigram.log 2>&1 &
```

**descriptions**

- `conf`：path fo configuration where we can specify the learning rate, batch_size, dropout rate ...
- `preprocess`： whether preprocess the training data, used by default in training phase and not used in testing phase.
- `device`： GPU id
- `marg`：whether use marginal probability of spans as the input of inside/CKY algorithm，used by default for all phases.
- `mask_inside`：whether exclude illegal trees in inside algorithm.
- `feat`: char representations which is input to BiLSTM choices：1) bichar embedding; 2) BERT
- `frain`：path to train data, must in pid format：

```
((IP (NP (NN 巧克力)) (VP (ADVP (AD 很)) (VP (VA 美味))) (PU .)))
```

- `fdev`：path to dev data.
- `ftest`：path to test data.
- `file`：directory to save model and fields (constructed from train data), named `model` and `fields` respectively.


## Test

```sh
python -u run.py predict \
    --conf=config.ini \
    --device=0   \
    --marg \
    --mask_inside \
    --mask_cky \
    --feat=bigram \
    --file=exp/ctb51.joint.bigram \
    --fdata=data/ctb51/test.pid \
    --fpred=exp/ctb51.joint.bigram/test.pid \
    > exp/ctb51.joint.bigram/test.log 2>&1
```

**most parameters are consistent with that in training phase, and other descriptions are as follows:**

- `marg`：consistent with the training phase.
- `mask_inside`：consistent with the training phase.
- `mask_cky`：whether exclude illegal trees in CKY algorithm.，used by default in testing phase and not used in training phase.
- `file`: ddirectory to save model and fields (constructed from train data), named `model` and `fields` respectively.
- `fdata`： data to be predicted, must in pid format:

```
((IP 巧克力很美味.)) # ((IP continue_chars))
```

- `fpred`： path to predicted results.

## Download models

The model of the joint framework is available at [`joint-ctb51`](http://hlt.suda.edu.cn/LA/yhou/CoNLL/joint-ctb51.tar.gz), [`joint-ctb51-big`](http://hlt.suda.edu.cn/LA/yhou/CoNLL/joint-ctb51-big.tar.gz), [`joint-ctb7`](http://hlt.suda.edu.cn/LA/yhou/CoNLL/joint-ctb7.tar.gz) and the model of the pipeline framework is available at [`pipeline-ctb51`](http://hlt.suda.edu.cn/LA/yhou/CoNLL/pipeline-ctb51.tar.gz), [`pipeline-ctb51-big`](http://hlt.suda.edu.cn/LA/yhou/CoNLL/pipeline-ctb51-big.tar.gz), [`pipeline-ctb7`](http://hlt.suda.edu.cn/LA/yhou/CoNLL/pipeline-ctb7.tar.gz).