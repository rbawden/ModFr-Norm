#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/../../../`
dataset=test
hyp_file=$thisdir/checkpoint_bestwordacc_sym.pt.$dataset.postproc
ref_file=$maindir/data/raw/$dataset/$dataset.finalised.trg
meta_file=$maindir/data/raw/$dataset/$dataset.finalised.meta
cache_file=$thisdir/.cache_lev/all.pickle

bash $maindir/scripts/eval.sh $thisdir $hyp_file $ref_file $meta_file $cache_file \
			 > $thisdir/checkpoint_bestwordacc_sym.pt.$dataset.eval

