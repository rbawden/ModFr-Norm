#!/bin/sh

modeldir=$1

# check args
if [ "$#" -ne 1 ]; then
    echo "Error: expected at least 1 argument: modeldir"
    echo "Usage: $0 <modeldir>"
    exit
fi

thisdir=`dirname $0`
maindir=`realpath $thisdir/../`
evalscriptsdir=$maindir/eval-scripts

ref=$maindir/data/raw/dev/dev.finalised.trg
meta=$maindir/data/raw/dev/dev.finalised.meta

if [ -f $modeldir/valid.eval ]; then
    echo "$modeldir/valid.eval already exists."
    echo "I am not going to rewrite it. Delete file to recalculate."
else
    for model in `ls -tr $modeldir/checkpoint*.pt`; do
	modelname=`basename $model`
	if [ $modelname != 'checkpoint_bestwordacc.pt' ] && \
	       [ $modelname != 'checkpoint_best.pt' ] && \
	       [ $modelname != 'checkpoint_last.pt' ]; then
	    num=`echo $model | perl -pe 's/^.*?checkpoint(\d+).pt/\1/'`
	    cachefile=$modeldir/.cache.pickle
	    # calculate scores
	    scores=`bash $evalscriptsdir/eval_detailed.sh $modeldir $ref $meta \
	    		  $modeldir/valid_outputs/$modelname.valid.postproc $cachefile`
	    echo -e "$model\t$scores" >> $modeldir/valid.eval
	fi
    done
fi
# plot the validation graph
python $thisdir/plot_val.py $modeldir/valid.eval $modeldir/valid_eval_plot.pdf

# get the best checkpoint according to word accuracy
best_checkpoint_num=`python $thisdir/select_best.py $modeldir/valid.eval wordacc_sym`
best_model=$modeldir/checkpoint$best_checkpoint_num.pt
echo $best_checkpoint_num
echo $best_model
cp $best_model $modeldir/checkpoint_bestwordacc_sym.pt
cp $modeldir/valid_outputs/checkpoint$best_checkpoint_num.pt.valid.postproc \
   $modeldir/checkpoint_bestwordacc_sym.pt.valid.postproc
cp -r $modeldir/valid_outputs/checkpoint$best_checkpoint_num.pt.valid.postproc-individual_files \
   $modeldir/checkpoint_bestwordacc_sym.pt.valid.postproc-individual_files

# write out the results line of the best model
cat $modeldir/valid.eval | \
    grep checkpoint$best_checkpoint_num.pt \
	 > $modeldir/checkpoint_bestwordacc_sym.eval
