#!/bin/sh

ref=$1
pred=$2
lang=$3

# check args
if [ "$#" -ne 3 ]; then
    echo "Error: expected 3 arguments: pred ref lang"
    echo "Usage: $0 <pred> <ref> <lang>"
    exit
fi

# default tokenisation apart from Chinese
tok=''
if [[ $lang == 'zh' ]]; then
    lang='-tok zh'
fi

# print with score only (option -b)
cat $pred | sacrebleu -b $ref $tok --width 4
