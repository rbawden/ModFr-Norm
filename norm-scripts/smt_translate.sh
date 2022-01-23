#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/../`
tools=$TOOLDIR # CHANGE THIS PATH AS NECESSARY TO THE DIRECTORY WHERE YOUR TOOLS ARE INSTALLED
kenlm=$tools/kenlm
fastalign=$tools/fast_align/build
moses=$tools/mosesdecoder
mgiza=$tools/mgiza/mgizapp/build

model_folder=$1

segtype=bpe_joint_500
rand=$RANDOM

# check args
if [ "$#" -ne 1 ]; then
    echo "Error: expected 2 arguments: model_folder"
    echo "Usage: $0 <model_folder>"
    exit
fi

python $maindir/data-scripts/spm_encode.py \
       --model=$maindir/data/preprocessed/$segtype.model \
       --output_format=piece \
       --outputs=$model_folder/tmp.$rand

# replace <main_folder> by absolute link
cp $model_folder/moses-tuned.ini $model_folder/moses-tuned-orig.ini
cat $model_folder/moses-tuned-orig.ini | perl -pe "s|path=<main_folder>|path=$maindir|g" > $model_folder/moses-tuned.ini


# first filter the phrase table
if [ ! -f $model_folder/filtered-tmp.$rand/moses-tuned.ini ]; then
    $moses/scripts/training/filter-model-given-input.pl \
	$model_folder/filtered-tmp.$rand \
	$model_folder/moses-tuned.ini $model_folder/tmp.$rand
fi

# then translate
$moses/bin/moses -f $model_folder/filtered-tmp.$rand/moses.ini -i $model_folder/tmp.$rand | \
    perl -pe 's/ //g' | perl -pe 's/▁/ /g'

cp $model_folder/moses-tuned-orig.ini $model_folder/moses-tuned.ini
rm $model_folder/tmp.$rand
rm -r $model_folder/filtered-tmp.$rand
