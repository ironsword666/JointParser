#! /bin/bash


data="ctb51"
device=0
method="joint"
feat="bigram"

nohup python -u run.py train \
        -p \
        -d=$device \
        --marg \
        --mask_inside \
        --feat=$feat \
        --ftrain=data/$data/train.pid \
        --fdev=data/$data/dev.pid \
        --ftest=data/$data/test.pid \
        --file=exp/$data.$method.$feat \
        > log/$data.$method.$feat 2>&1 &






