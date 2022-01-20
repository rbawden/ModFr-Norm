#!/bin/sh
thisdir=`dirname $0`
maindir=$thisdir/../../../
data=$maindir/data/shed/bin/bpe_joint_1000
model=checkpoint_bestwordacc_sym.pt
dataset=test

bash $maindir/scripts/translate.sh $model $dataset $data $thisdir/$model.$dataset

