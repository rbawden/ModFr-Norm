#!/bin/sh

thisdir=`dirname $0`
maindir=$thisdir/..
kenlm=~/tools/kenlm # change this path as necessary
outputdir=$maindir/lms/kenlm
[ -d $maindir/lms ] || mkdir $maindir/lms
[ -d $outputdir ] || mkdir $outputdir


n=4 # Specify value of n for n-gram models

# train language models on the target side of the parallel data + the monolingual contemporary French data from FreEm max.
for seg in char bpe_joint_500 bpe_joint_1000 bpe_joint_2000; do
    cat $maindir/data/preprocessed/mono+para/dalembert+train.char.trg | \
	$kenlm/build/bin/lmplz -o ${n} --discount_fallback > $outputdir/dalembert+train_${seg}_${n}g.trg
done


# train language models just on the tareget side of the parallel data
for n in 3 4 5; do
    for seg in char bpe_joint_500 bpe_joint_1000 bpe_joint_2000 bpe_joint_4000 bpe_joint_8000 bpe_joint_16000 bpe_joint_24000; do
	cat $maindir/data/preprocessed/train/train.$seg.trg | $kenlm/build/bin/lmplz -o ${n} --discount_fallback > $outputdir/para_${seg}_${n}g.trg
    done
done
