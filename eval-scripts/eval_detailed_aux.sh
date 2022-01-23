#!/bin/sh

thisdir=`dirname $0`

ref_file=$1
hyp_file=$2
type=$3
cache_file=$4

# check args
if [ "$#" -lt 3 ]; then
    echo "Error: expected 3 arguments: hyp ref type (cache_file)"
    echo "Usage: $0 <hyp> <ref> <type> (<cache_file>)"
    exit
fi

if [ ! -z $cache_file ]; then
    cache_file="--cache $cache_file"
fi

# List of different evaluation scripts
bleu_score=`bash $thisdir/bleu.sh $ref_file $hyp_file fr`
chrf_score=`bash $thisdir/chrf.sh $ref_file $hyp_file fr`
lev_char_score=`python $thisdir/levenshtein.py $ref_file $hyp_file $cachefile`
wordacc_scores=`python $thisdir/word_acc.py $ref_file $hyp_file $cache_file -a "ref,pred,both" | perl -CS -Mutf8 -pe "s/^/§$type,wordacc_r2h=/; s/ /§$type,wordacc_h2r=/; s/ /§$type,wordacc_sym=/; s/§/ /g"`

printf "$type,bleu=$bleu_score $type,chrf=$chrf_score $type,lev_char=$lev_char_score$wordacc_scores"
