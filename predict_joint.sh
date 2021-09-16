#! /bin/bash

# parameters
data="ctb51"
feat="bigram"
device=0
method="joint"
testmethod="mask_inside.mask_cky"


echo "\nJoint"
mkdir -p exp/$data.$method.$feat.$testmethod

python -u run.py predict \
    -d=$device \
    --marg \
    --mask_inside \
    --mask_cky \
    --feat=$feat \
    -f=exp/$data.$method.$feat \
    --fdata=data/$data/test.pid \
    --fpred=exp/$data.$method.$feat.$testmethod/test.pid \
    > exp/$data.$method.$feat.$testmethod/test.log 2>&1



