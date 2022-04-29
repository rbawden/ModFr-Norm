#!/bin/sh

output_folder=$1 # e.g. final_outputs/best_lstm
ref_file=$2
train_file=$3
cache_file=$4

# check args
if [ "$#" -lt 3 ]; then
    echo "Error: expected 3 or 4 arguments: output_folder ref_file train_trg_file (cache_file)"
    echo "Usage: $0 <output_folder> <ref_file> <train_trg_file> (<cache>)"
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
ref_word_acc_oov=""
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
    new_ref_wordacc_oov=`python $thisdir/word_acc_oov.py $ref_file $pred_file $train_file -a "ref" $cache_file`
    ref_wordacc_oov="$new_ref_wordacc_oov $ref_wordacc_oov"
done




output_type="wiki" # or latex
sep="&"
if [[ $output_type == "wiki" ]]; then
    sep="|" # or &
    plain_text='--plain_text'
fi
avg_bleu=`python eval-scripts/avg.py "$bleu" $plain_text`
avg_chrf=`python eval-scripts/avg.py "$chrf" $plain_text`
avg_ref_wordacc=`python eval-scripts/avg.py "$ref_wordacc" $plain_text`
avg_sym_wordacc=`python eval-scripts/avg.py "$sym_wordacc" $plain_text`
avg_ref_wordacc_oov=`python eval-scripts/avg.py "$ref_wordacc_oov" $plain_text`
avg_lev=`python eval-scripts/avg.py "$lev" $plain_text`

echo -e "WordAcc (ref) $set WordAcc (sym) $set  WordAcc OOV (ref) $sep Levenshtein $sep BLEU $sep ChrF"
echo '-----'

echo -e "$avg_ref_wordacc $sep $avg_sym_wordacc $sep $avg_ref_wordacc_oov $sep $avg_lev $sep $avg_bleu $sep $avg_chrf"

