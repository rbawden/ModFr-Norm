#!/bin/sh
thisdir=`dirname $0`
maindir=`realpath $thisdir/../../../`
data=$maindir/data/shed/bin/bpe_joint_1000

model=$thisdir/checkpoint_bestwordacc_sym.pt
dataset=test

# translate the test set
bash $maindir/scripts/translate.sh $model $dataset $data $thisdir/$model.$dataset

