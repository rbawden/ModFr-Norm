#!/bin/sh
thisdir=`dirname "$(realpath $0)"` # define all locations wrt this folder
maindir=`realpath $thisdir/../../..`
data=$maindir/data/bin/bpe_joint_1000

# transformer
seed=2
arch=lstm
encoderlayers=3
decoderlayers=3
attentionheads=4
embeddim=384
ffdim=1024
dropout=0.3
lr=0.001
bsz=3000

# for RNN
hidden=768


if [[ $arch == "transformer" ]]; then
    
    fairseq-train \
	$data \
	--save-dir $thisdir --save-interval 5 --patience 25 --seed $seed --arch transformer \
	--encoder-layers $encoderlayers --decoder-layers $decoderlayers --encoder-attention-heads $attentionheads \
	--encoder-embed-dim $embeddim --encoder-ffn-embed-dim $ffdim --dropout $dropout \
	--criterion cross_entropy --optimizer adam --adam-betas '(0.9, 0.98)' --lr $lr --lr-scheduler inverse_sqrt \
	--warmup-updates 4000 --max-tokens 3000 --max-tokens $bsz \
	--share-all-embeddings --batch-size-valid 64

else
    fairseq-train \
        $data \
        --save-dir $thisdir --save-interval 5 --patience 25 --seed $seed --arch lstm \
        --encoder-layers $encoderlayers --decoder-layers $decoderlayers \
        --encoder-embed-dim $embeddim --decoder-embed-dim $embeddim --decoder-out-embed-dim $embeddim \
	--encoder-hidden-size $hidden --encoder-bidirectional --decoder-hidden-size $hidden \
	--dropout $dropout \
        --criterion cross_entropy --optimizer adam --adam-betas '(0.9, 0.98)' --lr $lr --lr-scheduler inverse_sqrt \
        --warmup-updates 4000 --max-tokens 3000 \
        --share-all-embeddings --batch-size-valid 64

fi
