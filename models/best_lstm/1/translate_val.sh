#!/bin/sh
thisdir=`dirname $0`
maindir=`realpath $thisdir/../../../`
valid_set=$maindir/data/bin/bpe_joint_1000

[ -d $thisdir/valid_outputs ] || mkdir $thisdir/valid_outputs
[ ! -f $thisdir/valid.eval ] || rm $thisdir/valid.eval

for model in `ls -tr checkpoint*.pt`; do
    # translate the valid set
    if [ ! -f $thisdir/valid_outputs/$model.valid ]; then
	bash $maindir/scripts/translate.sh $model 'valid' \
	     $valid_set $thisdir/valid_outputs/$model.valid
    fi
done
