#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/..`

dropout=0.3
lr=0.001 
batchsize=3000

arch=transformer
for seed in 1 2 3; do 
    for datatype in char bpe_joint_1000 char bpe_joint_500 bpe_joint_1000 bpe_joint_2000 bpe_joint_4000 bpe_joint_8000 bpe_joint_16000 bpe_joint_24000 ; do
	datapath=`realpath $maindir/data/bin/$datatype`
	for modelsize in medium; do

	    if [[ $arch == transformer ]]; then
		case $modelsize in
		    small) encoderlayers=2; decoderlayers=2; attnheads=2; embeddim=128; ffdim=512 ;;
		    medium) encoderlayers=4; decoderlayers=4; attnheads=4; embeddim=256; ffdim=1024 ;;
		    big) encoderlayers=6; decoderlayers=6; attnheads=8; embeddim=512; ffdim=2048 ;;
		    xsmall) continue;
		esac
		modeldir=`realpath $maindir/mt-models/${arch}_${datatype}_${encoderlayers}enc_${decoderlayers}dec_${attnheads}heads_${embeddim}embdim_${ffdim}ff_${dropout}drop_${lr}lr_${batchsize}bsz/`
	    elif [[ $arch == lstm ]]; then
		attnheads=None
		ffdim=None
	       case $modelsize in
                   xsmall) 
		       encoderlayers=1; decoderlayers=1; embeddim=128; hidden=256 ;;
		   small) 
		       encoderlayers=2; decoderlayers=2; embeddim=256; hidden=512 ;;
                   medium)
		       encoderlayers=3; decoderlayers=3; embeddim=384; hidden=768 ;;
                   big)
		       encoderlayers=4; decoderlayers=4; embeddim=512; hidden=1024 ;;
	       esac
	       modeldir=`realpath $maindir/mt-models/${arch}_${datatype}_${encoderlayers}enc_${decoderlayers}dec_${hidden}hidden_${embeddim}embdim_${dropout}drop_${lr}lr_${batchsize}bsz/`
	    else
		break
	    fi
	    
	    if [ ! -d $modeldir/$seed ]; then
		bash $thisdir/create_experiment.sh --datatype $datatype --datapath $datapath --modeldir $modeldir \
		     --arch $arch --encoderlayers $encoderlayers --decoderlayers $decoderlayers \
		     --attnheads $attnheads --embeddim $embeddim --ffdim $ffdim --dropout $dropout \
		     --lr $lr --seed $seed --hidden $hidden --bsz $batchsize
		
	    fi
	done
    done
done
