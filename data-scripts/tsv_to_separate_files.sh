#!/bin/sh

raw_folder=$1

# check args
if [ "$#" -ne 1 ]; then
    echo "Error: expected 1 argument: raw_folder"
    echo "Usage: $0 <raw_folder>"
    exit
fi

# split each file into src, trg and meta info
for setname in train dev test; do
    orig_file=$raw_folder/$setname/$setname.tsv
    out_prefix=$raw_folder/$setname/$setname
    cat $orig_file | cut -f1 > $out_prefix.src
    cat $orig_file | cut -f2 > $out_prefix.trg
    cat $orig_file | cut -f3- > $out_prefix.meta

    # create files for each subcorpus for dev/test
#    if [ $setname != train ]; then
#	subcorpora=`cat $orig_file | cut -f13 | sort -u`
#	for subcorpus in $subcorpora; do
#	    if [[ $subcorpus =~ [1-8] ]]; then
#		cat $orig_file | grep $subcorpus > $out_prefix.$subcorpus.tsv
#		cat $orig_file | grep $subcorpus | cut -f1 > $out_prefix.$subcorpus.src
#		cat $orig_file | grep $subcorpus | cut -f2 > $out_prefix.$subcorpus.trg
#		cat $orig_file | grep $subcorpus | cut -f3- > $out_prefix.$subcorpus.meta
#	    fi
#	done
 #   fi
done
