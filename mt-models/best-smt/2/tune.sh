#!/bin/sh

thisdir=`realpath $(dirname $0)`
maindir=`realpath $thisdir/../../..`
tools=$TOOLDIR # UPDATE THIS TO YOUR TOOL DIRECTORY 
kenlm=$tools/kenlm
fastalign=$tools/fast_align/build
moses=$tools/mosesdecoder
mgiza=$tools/mgiza/mgizapp/build

type=bpe_joint_500 # change this as appropriate
traindata=$maindir/data/preprocessed/train/train.$type
devdata=$maindir/data/preprocessed/dev/dev.$type

[ -d $thisdir/tuning ] || mkdir $thisdir/tuning



perl $moses/scripts/training/mert-moses.pl \
     $devdata.src $devdata.trg \
     $moses/bin/moses \
     $thisdir/model/moses.ini \
     --batch-mira \
     --working-dir $thisdir/tuning \
     --decoder-flags="-threads 8" \
     --nbest 25 \
     --mertdir $moses/bin \
     --return-best-dev \
     --maximum-iterations 25 2> $thisdir/tuning/tuning.log

# copy tuned moses file
cp $thisdir/tuning/moses.ini $thisdir/model/moses-tuned.ini
