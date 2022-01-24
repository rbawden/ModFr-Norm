#!/bin/sh

output_folder=$1 # e.g. final_outputs/best_lstm
ref_file=$2
cache_file=$3

# check args
if [ "$#" -lt 2 ]; then
    echo "Error: expected 2 or 3 arguments: output_folder ref_file (cache_file)"
    echo "Usage: $0 <output_folder> <ref_file> (<cache>)"
    exit
fi

thisdir=`dirname $0`
if [ ! -z $cache_file ]; then
    cache_file="--cache $cache_file"
fi

# get all scores
bleu=""
chrf=""
lev=""
ref_wordacc=""
sym_wordacc=""
for pred_file in `ls $output_folder/*trg 2>/dev/null`; do
    new_bleu=`bash $thisdir/bleu.sh $ref_file $pred_file fr`
    bleu="$bleu $new_bleu"
    new_chrf=`bash $thisdir/chrf.sh $ref_file $pred_file fr`
    chrf="$chrf $new_chrf"
    new_lev=`python $thisdir/levenshtein.py $ref_file $pred_file -a ref $cache_file`
    lev="$lev $new_lev"
    new_wordacc=`python $thisdir/word_acc.py $ref_file $pred_file -a "ref,both" $cache_file`
    new_sym_wordacc=`echo $new_wordacc | cut -d" " -f1`
    new_ref_wordacc=`echo $new_wordacc | cut -d" " -f2`
    ref_wordacc="$new_ref_wordacc $ref_wordacc"
    sym_wordacc="$new_sym_wordacc $sym_wordacc"
done


echo -e "WordAcc (ref) & WordAcc (sym) & Levenshtein & BLEU & ChrF \\"
echo '-----'

avg_bleu=`python eval-scripts/avg.py "$bleu"`
avg_chrf=`python eval-scripts/avg.py "$chrf"`
avg_ref_wordacc=`python eval-scripts/avg.py "$ref_wordacc"`
avg_sym_wordacc=`python eval-scripts/avg.py "$sym_wordacc"`
avg_lev=`python eval-scripts/avg.py "$lev"`
echo -e "$avg_ref_wordacc & $avg_sym_wordacc & $avg_lev & $avg_bleu & $avg_chrf \\\\\\"

