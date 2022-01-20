#!/bin/sh

raw_folder=$1

# check args
if [ "$#" -ne 1 ]; then
    echo "Error: expected 1 argument: raw_folder"
    echo "Usage: $0 <raw_folder>"
    exit
fi

# split each dev and test files into separate components
for setname in dev test; do
    prefix=$raw_folder/$setname/$setname.finalised
    subcorpora=`cat $prefix.meta | cut -f11 | sort -u`
    out_prefix=$raw_folder/$setname/$setname.finalised
    for subcorpus in $subcorpora; do

        if [[ $subcorpus =~ [1-8] ]]; then
	    paste $prefix.src $prefix.trg $prefix.meta | grep $subcorpus > $out_prefix.$subcorpus.tsv
	    paste $prefix.src $prefix.trg $prefix.meta | grep $subcorpus | cut -f1 > $out_prefix.$subcorpus.src
	    paste $prefix.src $prefix.trg $prefix.meta | grep $subcorpus | cut -f2 > $out_prefix.$subcorpus.trg
	    paste $prefix.src $prefix.trg $prefix.meta | grep $subcorpus | cut -f3- > $out_prefix.$subcorpus.meta
        fi
    done
    
done
