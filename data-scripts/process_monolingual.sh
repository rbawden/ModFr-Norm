#!/bin/sh

thisdir=`dirname "$(realpath $0)"` # define all locations wrt the script folder                                                                                                                              
. $thisdir/tool_paths.sh # load tool paths

train_files="$thisdir/../data/raw/train/train.finalised.src,$thisdir/../data/raw/train/train.finalised.trg"
dalembert=$thisdir/../../dAlemBERT
preproc_dir=$thisdir/../data/preprocessed/

# create preprocessing folder
[ -d $preproc_dir/mono ] || mkdir $preproc_dir/mono

if true; then
# get raw data (and metadata) from dAlembert
python scripts/get_monolingual_normalised.py $dalembert/3_TXT/ $dalembert/TOC.tsv -u > $thisdir/../data/raw/mono/dAlemBERT.src.tsv
python scripts/get_monolingual_normalised.py $dalembert/3_TXT/ $dalembert/TOC.tsv > $thisdir/../data/raw/mono/dAlemBERT.trg.tsv

# divide into different files
cat $thisdir/../data/raw/mono/dAlemBERT.src.tsv | cut -f1 > $thisdir/../data/raw/mono/dAlemBERT.src
cat $thisdir/../data/raw/mono/dAlemBERT.src.tsv | cut -f2- > $thisdir/../data/raw/mono/dAlemBERT.src.meta
cat $thisdir/../data/raw/mono/dAlemBERT.trg.tsv | cut -f1 > $thisdir/../data/raw/mono/dAlemBERT.trg
cat $thisdir/../data/raw/mono/dAlemBERT.trg.tsv | cut -f2- > $thisdir/../data/raw/mono/dAlemBERT.trg.meta

# Pre-normalisation (i.e. homogenise certain typographic variants such as apostrophes, quotes, ligatures, double consonants, etc.)
echo ">>> Pre-normalise certain typograhpics variants (quote marks and apostrophes)"
for lang in src trg; do
    cat $thisdir/../data/raw/mono/dAlemBERT.$lang | bash $thisdir/normalise.sh > $thisdir/../data/raw/mono/dAlemBERT.norm.$lang
done

# Deduplicate
for lang in src trg; do
    paste $thisdir/../data/raw/mono/dAlemBERT.norm.$lang $thisdir/../data/raw/mono/dAlemBERT.$lang.meta | \
	python $thisdir/../scripts/deduplicate.py > $thisdir/../data/raw/mono/dAlemBERT.norm.dedup.$lang.tsv
    cat $thisdir/../data/raw/mono/dAlemBERT.norm.dedup.$lang.tsv | cut -f1 > $thisdir/../data/raw/mono/dAlemBERT.norm.dedup.$lang
    cat $thisdir/../data/raw/mono/dAlemBERT.norm.dedup.$lang.tsv | cut -f2- > $thisdir/../data/raw/mono/dAlemBERT.norm.dedup.$lang.meta
done
fi
# Remove any identical test sentences from the train and dev sets (either side of the data)
# TODO

final=dAlemBERT.norm.dedup

# apply the models
for lang in src trg; do
    for model_prefix in bpe_joint_1000; do 
	#bpe_joint_500 bpe_joint_1000 bpe_joint_2000 bpe_joint_4000 bpe_joint_8000 bpe_joint_16000 bpe_joint_24000 char; do
	
	cat $thisdir/../data/raw/mono/$final.$lang | \
	    bash $thisdir/encode_spm.sh $preproc_dir/$model_prefix.model > $preproc_dir/mono/$final.$model_prefix.$lang
    done
done


