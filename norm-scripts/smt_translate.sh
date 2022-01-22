#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/../`
tools=/gpfswork/rech/ncm/commun/tools # CHANGE PATH AS NECESSARY
kenlm=$tools/kenlm
fastalign=$tools/fast_align/build
moses=$tools/mosesdecoder
mgiza=$tools/mgiza/mgizapp/build

source=$1
model_folder=$2

# check args
if [ "$#" -ne 2 ]; then
    echo "Error: expected 2 arguments: input model_folder"
    echo "Usage: $0 <input> <model_folder>"
    exit
fi

# replace <main_folder> by absolute link
cp $model_folder/moses-tuned.ini $model_folder/moses-tuned-orig.ini
cat $model_folder/moses-tuned-orig.ini | perl -pe "s|path=<main_folder>|path=$maindir|g" > $model_folder/moses-tuned.ini


# first filter the phrase table
src_base="$(basename -- $source)"
if [ ! -f $model_folder/filtered-$src_base/moses-tuned.ini ]; then
    echo "$moses/scripts/training/filter-model-given-input.pl $model_folder/filtered-$src_base $model_folder/moses-tuned.ini $source"
    $moses/scripts/training/filter-model-given-input.pl $model_folder/filtered-$src_base $model_folder/moses-tuned.ini $source
fi

# then translate
$moses/bin/moses -f $model_folder/filtered-$src_base/moses.ini -i $source | \
    perl -pe 's/ //g' | perl -pe 's/▁/ /g'

cp $model_folder/moses-tuned-orig.ini $model_folder/moses-tuned.ini
