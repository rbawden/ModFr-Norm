#!/bin/sh

thisdir=`dirname $0`

modeldir=$1
ref_file=$1
meta_file=$2
hyp_file=$3
cache_file=$5

# check args
if [ "$#" -lt 4 ]; then
    echo "Error: expected 4 (or 5) arguments: modeldir hyp_file ref_file meta_file (cache_file)"
    echo "Usage: $0 <modeldir> <hyp_file> <ref_file> <meta_file> (<cache_file>)"
    exit
fi

# evaluate on the whole test set
scores=`bash $thisdir/eval_detailed_aux.sh $ref_file $hyp_file all $cache_file`
hyp_name=`basename $hyp_file`

# divide set into separate files to evaluate separate genres
folder=$hyp_file-individual_files/
[ -d $folder ] || mkdir $folder
python $thisdir/divide_test_set_for_eval.py $hyp_file $meta_file $folder/$hyp_name

# evaluate each one
for hyp_file_spec in `ls $folder/$hyp_name*`; do
    subcorpus=${hyp_file_spec##*.}
    # get basename of ref without extension and add subcorpus name
    ref_file_dir=`dirname $ref_file`
    ref_file_basename=`basename $ref_file`
    ref_file_ext=${ref_file_basename##*.}
    ref_file_basename=`echo $ref_file_basename | cut -f1,2 -d"."`
    ref_file_spec=$ref_file_dir/$ref_file_basename.$subcorpus.$ref_file_ext
    tmp_scores=`bash $thisdir/eval_detailed_aux.sh $ref_file_spec $hyp_file_spec $subcorpus $cache_file`
    scores="$scores $tmp_scores"
done

# output final scores
echo $scores
