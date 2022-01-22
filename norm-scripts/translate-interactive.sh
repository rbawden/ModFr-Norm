#!/bin/sh
model_path="$1" # path to the model file (.pt)
dataset=$2 # name of dataset (train, valid or test)
data=$3 # path to binarised data folder (from fairseq-preprocess)
output=$4 # output file prefix

# check args
if [ "$#" -ne 4 ]; then
    echo "Error: expected 4 arguments: model_path data_path input output"
    echo "Usage: $0 <model_path> <data_path> <input> <output>"
    exit
fi

if [ ! -f $output.output ]; then
    cat $datafile | fairseq-interactive $data --path $model_path > $output.output
fi

if [ ! -f $output.postproc ]; then
    cat $output.output | grep "H-" | perl -pe 's/^H-//' | sort -n | cut -f3 | perl -pe 's/ //g;s/▁/ /g' | perl -pe 's/<decade=.+?>//g' > $output.postproc
fi




