data="ctb51"
feat="bigram"
method="joint"
testmethod="mask_inside.mask_cky"

echo $data

echo "\ntest"
python .evalp.py -p=exp/$data.joint.$method.$cwsfeat.$testmethod/test.pid -g=data/$data/test.pid -t
python .evalb.py -p=exp/$data.joint.$method.$cwsfeat.$testmethod/test.pid -g=data/$data/test.pid