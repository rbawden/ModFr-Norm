#!/bin/sh

# define all locations wrt the script folder
thisdir=`dirname "$(realpath $0)"`
maindir=`realpath $thisdir/..`
preproc_dir=$maindir/data/preprocessed/
bin_dir=$maindir/data/bin
inv_bin_dir=$maindir/data/bin-opposite
camembert=/home/rbawden/scratch/lms/camembert/camembert-base
dalembert_path=/home/rbawden/scratch/lms/dalembert/epoch_41/

# 1. Prepare basic data (text only)
echo ">>> Training BPE models"
[ -d $preproc_dir ] || mkdir -p $preproc_dir
train_files="$maindir/data/raw/train/train.finalised.src,$maindir/data/raw/train/train.finalised.trg"
# train different models of different vocab size
for vocab_size in 500 1000 2000 4000 8000 16000 24000; do
    if [ ! -f $preproc_dir/bpe_joint_$vocab_size.model ]; then
	python $thisdir/spm_train.py \
	       --input=$train_files \
	       --model_prefix=$preproc_dir/bpe_joint_$vocab_size \
	       --model_type=bpe \
	       --vocab_size=$vocab_size \
	       --character_coverage=1.0
    fi
done

echo ">>> Training character models"
# train a character model
if [ ! -f $preproc_dir/char.model ]; then
    python $thisdir/spm_train.py \
               --input=$train_files \
               --model_prefix=$preproc_dir/char \
               --model_type=char \
               --character_coverage=1.0
fi

# dalembert
for dataset in train dev test; do
    for lang in src trg; do
	cat $maindir/data/raw/$dataset/$dataset.finalised.$lang | \
	    python $thisdir/hf_tokenise.py $dalembert_path/tokenizer.json \
	    > $preproc_dir/$dataset/$dataset.dalembert.$lang
    done
done

# create preprocessing files
for dataset in train dev test; do
    [ -d $preproc_dir/$dataset ] || mkdir $preproc_dir/$dataset
done

echo ">>> Applying BPE + char models"
[ -d $bin_dir ] || mkdir -p $bin_dir
# apply the models
for model_prefix in bpe_joint_500 bpe_joint_1000 bpe_joint_2000 \
				  bpe_joint_4000 bpe_joint_8000 \
				  bpe_joint_16000 bpe_joint_24000 char camembert dalembert; do
    for dataset in train dev test; do
	for lang in src trg; do
	    if [ ! -f $preproc_dir/$dataset/$dataset.$model_prefix.$lang ]; then
		cat $maindir/data/raw/$dataset/$dataset.finalised.$lang | \
		    python $thisdir/spm_encode.py \
			   --model=$preproc_dir/$model_prefix.model \
			   --output_format=piece \
			   --outputs=$preproc_dir/$dataset/$dataset.$model_prefix.$lang

	    fi
	done
    done
    echo ">> Binarising"
    ditionary="--joined-dictionary"
    if [ $model_prefix == camembert ]; then
	dictionary="--srcdict $camembert/dict.txt --tgtdict $camembert/dict.txt"
    elif [ $model_prefix == dalembert ]; then
	dictionary="--srcdict $dalembert_path/dict-without-symbols.txt --tgtdict $dalembert_path/dict-without-symbols.txt"
    fi
    # binarise data in fairseq format
    if [ ! -d $bin_dir/$model_prefix ]; then
	fairseq-preprocess --destdir $bin_dir/$model_prefix -s src -t trg \
			   --trainpref $preproc_dir/train/train.$model_prefix \
			   --validpref $preproc_dir/dev/dev.$model_prefix \
			   --testpref $preproc_dir/test/test.$model_prefix \
			   $dictionary
    fi
done

# 2. prepare data with meta information

# decade and text
for model_prefix in bpe_joint_1000; do
    for dataset in train dev test; do
	# add a meta token to the data
	for lang in src trg; do
	    paste <(cat data/raw/$dataset/$dataset.finalised.meta | \
			cut -f3 | perl -pe 's/[\[\]]//g' | cut -c1-3) \
		  $preproc_dir/$dataset/$dataset.$model_prefix.$lang \
		| perl -pe 's/^(\d\d\d)/▁<decade=\1> /' \
		| perl -pe 's/\t/ /' \
		       > $preproc_dir/$dataset/$dataset.$model_prefix.decade-token.$lang

	    paste <(cat data/raw/$dataset/$dataset.finalised.meta \
			| cut -f3 | perl -pe 's/[\[\]]//g' | cut -c1-3) \
		  <(cat data/raw/$dataset/$dataset.finalised.meta \
			| cut -f3 | perl -pe 's/[\[\]]//g' | cut -c4) \
                  $preproc_dir/$dataset/$dataset.$model_prefix.$lang  \
                | perl -pe 's/^(\d\d\d)/▁<decade=\1> /' \
		| perl -pe 's/\t(\d)/ ▁<year=\1>/' \
                       > $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens.$lang
	    
	    paste $preproc_dir/$dataset/$dataset.$model_prefix.$lang \
		  <(cat data/raw/$dataset/$dataset.finalised.meta \
			| cut -f3 | perl -pe 's/[\[\]]//g' | cut -c1-3) \
                | perl -pe 's/\t(\d\d\d)/ ▁<decade=\1> /' \
                       > $preproc_dir/$dataset/$dataset.$model_prefix.decade-end-token.$lang
	done
	cat data/raw/$dataset/$dataset.finalised.meta | cut -f3 \
	    | perl -pe 's/[\[\]]//g' | cut -c1-3 \
	    | perl -pe 's/(\d\d\d)/▁<decade=\1> /' \
		   > $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token
	
	# sym link to these datasets to get a token either on the input or output side of the data
	for typedec in "-" "-end-"; do
	    ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.src \
	       $preproc_dir/$dataset/$dataset.$model_prefix.decade${typedec}trg.src
	    ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.decade${typedec}token.trg \
	       $preproc_dir/$dataset/$dataset.$model_prefix.decade${typedec}trg.trg
	    ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.trg \
	       $preproc_dir/$dataset/$dataset.$model_prefix.decade${typedec}src.trg
	    ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.decade${typedec}token.src \
	       $preproc_dir/$dataset/$dataset.$model_prefix.decade${typedec}src.src
	done
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.src \
	   $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token-trg.src
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token \
	   $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token-trg.trg
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.trg \
	   $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token-src.trg
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token \
	   $preproc_dir/$dataset/$dataset.$model_prefix.just-decade-token-src.src
	# decade - year - represent as 2 tokens
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.src \
	   $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens-trg.src
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens.trg \
	   $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens-trg.trg
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.trg \
	   $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens-src.trg
	ln -sf $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens.src \
	   $preproc_dir/$dataset/$dataset.$model_prefix.year-2tokens-src.src
    done
    # binarise w/ decade token on src or trg
    for lang in src trg; do
	for typedec in "-" "-end-"; do
	    if [ ! -d $bin_dir/$model_prefix.decade${typedec}$lang ]; then
		fairseq-preprocess --destdir $bin_dir/$model_prefix.decade${typedec}$lang -s src -t trg \
				   --trainpref $preproc_dir/train/train.$model_prefix.decade${typedec}$lang \
				   --validpref $preproc_dir/dev/dev.$model_prefix.decade${typedec}$lang \
				   --testpref $preproc_dir/test/test.$model_prefix.decade${typedec}$lang \
				   --joined-dictionary
	    fi
	    if [ ! -d $inv_bin_dir/$model_prefix.decade${typedec}$lang ]; then
	    fairseq-preprocess --destdir $inv_bin_dir/$model_prefix.decade${typedec}$lang -s trg -t src \
                               --trainpref $preproc_dir/train/train.$model_prefix.decade${typedec}$lang \
                               --validpref $preproc_dir/dev/dev.$model_prefix.decade${typedec}$lang \
                               --testpref $preproc_dir/test/test.$model_prefix.decade${typedec}$lang \
			       --joined-dictionary
	    fi
	    if [ ! -d $inv_bin_dir/$model_prefix.decade${typedec}$lang ]; then
		fairseq-preprocess --destdir $inv_bin_dir/$model_prefix.decade${typedec}$lang -s trg -t src \
				   --trainpref $preproc_dir/train/train.$model_prefix.decade${typedec}$lang \
				   --validpref $preproc_dir/dev/dev.$model_prefix.just-decade-token$lang \
				   --testpref $preproc_dir/test/test.$model_prefix.decade${typedec}$lang
	    fi
	done
    done
    if [ ! -d $bin_dir/$model_prefix.year-2tokens-$lang ]; then
	fairseq-preprocess --destdir $bin_dir/$model_prefix.year-2tokens-trg -s src -t trg \
                           --trainpref $preproc_dir/train/train.$model_prefix.year-2tokens-trg \
                           --validpref $preproc_dir/dev/dev.$model_prefix.year-2tokens-trg \
                           --testpref $preproc_dir/test/test.$model_prefix.year-2tokens-trg \
			   --joined-dictionary
    fi
    if [ ! -d $inv_bin_dir/$model_prefix.year-2tokens-src ]; then
	fairseq-preprocess --destdir $inv_bin_dir/$model_prefix.year-2tokens-src -s trg -t src \
                           --trainpref $preproc_dir/train/train.$model_prefix.year-2tokens-src \
                           --validpref $preproc_dir/dev/dev.$model_prefix.year-2tokens-src \
                           --testpref $preproc_dir/test/test.$model_prefix.year-2tokens-src \
			   --joined-dictionary
    fi
    if [ ! -d $bin_dir/$model_prefix.just-decade-token-trg ]; then
	fairseq-preprocess --destdir $bin_dir/$model_prefix.just-decade-token-trg -s src -t trg \
			   --trainpref $preproc_dir/train/train.$model_prefix.just-decade-token-trg \
			   --validpref $preproc_dir/dev/dev.$model_prefix.just-decade-token-trg \
			   --testpref $preproc_dir/test/test.$model_prefix.just-decade-token-trg \
			   --joined-dictionary
    fi
    if [ ! -d $inv_bin_dir/$model_prefix.just-decade-token-src ]; then
	fairseq-preprocess --destdir $inv_bin_dir/$model_prefix.just-decade-token-src -s trg -t src \
			   --trainpref $preproc_dir/train/train.$model_prefix.just-decade-token-src \
			   --validpref $preproc_dir/dev/dev.$model_prefix.just-decade-token-src \
			   --testpref $preproc_dir/test/test.$model_prefix.just-decade-token-src \
			   --joined-dictionary
    fi
done

