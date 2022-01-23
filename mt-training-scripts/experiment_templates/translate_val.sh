#!/bin/sh
thisdir=`dirname $0`
maindir=$thisdir/../../../
valid_set=<datapath>

[ -d $thisdir/valid_outputs ] || mkdir $thisdir/valid_outputs
[ ! -f $thisdir/valid.eval ] || rm $thisdir/valid.eval

for model in `ls -tr checkpoint*.pt`; do
    # translate the valid set
    if [ ! -f $thisdir/valid_outputs/$model.valid.postproc ]; then
	bash $maindir/mt-training-scripts/translate-generate.sh $model 'valid' $valid_set $thisdir/valid_outputs/$model.valid
    fi
done
