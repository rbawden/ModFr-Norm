#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/..`

[ -d $maindir/final-mt-models ] || mkdir $maindir/final-mt-models

if [ ! -d $maindir/final-mt-models/lstm ]; then
    if [ ! -f $maindir/lstm.zip ]; then
	wget https://zenodo.org/record/6594765/files/lstm.zip?download=1
	mv $maindir/lstm.zip?download=1 $maindir/lstm.zip
    fi
    unzip $maindir/lstm.zip -d $maindir/final-mt-models/
else
    echo "The folder $maindir/final-mt-models/lstm already exists so not redownloading. To force the download of the LSTM model, please delete this folder first."
fi

if [ ! -d $maindir/final-mt-models/smt ]; then
    if [ ! -f $maindir/smt.zip ]; then
	wget https://zenodo.org/record/6594765/files/smt.zip?download=1
	mv $maindir/smt.zip?download=1 $maindir/smt.zip
    fi
    unzip $maindir/smt.zip -d $maindir/final-mt-models/
else
    echo "The folder $maindir/final-mt-models/smt already exists so not redownloading. To force the download of the SMT model, please delete this folder first."
fi

if [ ! -d $maindir/final-mt-models/transformer ]; then
    if [ ! -f $maindir/transformer.zip ]; then
	wget https://zenodo.org/record/6594765/files/transformer.zip?download=1
	mv $maindir/transformer.zip?download=1 $maindir/transformer.zip
    fi
    unzip $maindir/transformer.zip -d $maindir/final-mt-models/
else
    echo "The folder $maindir/final-mt-models/transformer already exists so not redownloading. To force the download of the transformer model, please delete this folder first."
fi


