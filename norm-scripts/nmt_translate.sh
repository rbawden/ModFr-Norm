#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/../`
model_path="$1" # path to the model file (.pt)

# check args
if [ "$#" -ne 1 ]; then
    echo "Error: expected 2 arguments: model_folder"
    echo "Usage: $0 <model_folder>"
    exit
fi

model_folder=`dirname $model_path`
# deduce segmentation from the train script
segtype=`cat $model_folder/train.sh | grep 'data=\$maindir' | perl -pe "s|^data=.maindir/data/bin/||"`
data=$maindir/data/bin/$segtype

rand=$RANDOM
python $maindir/data-scripts/spm_encode.py \
       --model=$maindir/data/preprocessed/$segtype.model \
       --output_format=piece \
       --outputs=$model_folder/tmp.$rand

cat $model_folder/tmp.$rand | fairseq-interactive $data --path $model_path | \
    grep "H-" | perl -pe 's/^H-//' | sort -n | cut -f3 | perl -pe 's/ //g;s/▁/ /g'

rm $model_folder/tmp.$rand





