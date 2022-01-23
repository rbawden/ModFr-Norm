#!/bin/sh
thisdir=`dirname $0`
maindir=`realpath $thisdir/../../..`
tools=~tools # change location of your tools as required
kenlm=$tools/kenlm
fastalign=$tools/fast_align/build
moses=$tools/mosesdecoder
mgiza=$tools/mgiza/mgizapp/build

type=bpe_joint_500 # change segmentation type as required
traindata=$maindir/data/preprocessed/train/train.$type

[ -d $thisdir/model ] || mkdir $thisdir/model
[ -d $thisdir/alignments ] || mkdir $thisdir/alignments

# prepare align
paste $traindata.src $traindata.trg | perl -pe 's/\t/ \|\|\| /' > $thisdir/alignments/train.prepare_align
# align forward
if [ ! -f $thisdir/alignments/train.forward ]; then
    $fastalign/fast_align -i $thisdir/alignments/train.prepare_align -d -o -v > $thisdir/alignments/train.forward
fi
# align backward
if [ ! -f $thisdir/alignments/train.backward ]; then
    $fastalign/fast_align -i $thisdir/alignments/train.prepare_align -d -o -v -r > $thisdir/alignments/train.backward
fi
# symmetrise
if [ ! -f $thisdir/model/aligned.grow-diag-final-and ]; then
    $fastalign/atools -i $thisdir/alignments/train.forward -j $thisdir/alignments/train.backward -c grow-diag-final-and > $thisdir/model/aligned.grow-diag-final-and
fi

# step 1 of model training
$moses/scripts/training/train-model.perl --cores 8 --first-step 1 --last-step 1 \
                                         --external-bin-dir $mgiza --mgiza-cpus 8 \
                                         --root-dir $thisdir --corpus $traindata \
                                         -f src -e trg --alignment grow-diag-final-and \
                                         --reordering msd-bidirectional-fe \
                                         -lm 0:4:$maindir/lms/kenlm/para_${type}_4g.trg.arpa.bin 2> $thisdir/model/training.1-1.log
# steps 4-9 of model training
$moses/scripts/training/train-model.perl --cores 8 --first-step 4 --last-step 9 \
                                         --external-bin-dir $mgiza --mgiza-cpus 8 \
                                         --root-dir $thisdir --corpus $traindata \
                                         -f src -e trg --alignment grow-diag-final-and \
                                         --reordering msd-bidirectional-fe \
                                         -lm 0:4:$maindir/lms/kenlm/para_${type}_4g.trg.arpa.bin 2> $thisdir/model/training.4-9.log
