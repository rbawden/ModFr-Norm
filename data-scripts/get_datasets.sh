#!/bin/sh

thisdir=`dirname "$(realpath $0)"` # define all locations wrt the script folder
maindir=`realpath $thisdir/..`
datapath=$maindir/data/raw/
raw_folder=$maindir/data/FreEM-corpora-FreEMnorm-9379caf

# Get dataset splits (train, dev, test)
[ -d $maindir/data ] || mkdir -p $maindir/data/raw
# Output to FreEM-corpora-FreEMnorm-9379caf/split/{train,dev,test}/
if [ ! -d $raw_folder ]; then
    wget https://zenodo.org/record/5865428/files/FreEM-corpora/FreEMnorm-1.0.0.zip
    mv $maindir/FreEMnorm-1.0.0.zip $maindir/data/ && \
	unzip $maindir/data/FreEMnorm-1.0.0.zip -d $maindir/data/ && \
	rm $maindir/data/FreEMnorm-1.0.0.zip
fi

# Create folders
for dataset in train dev test; do
    [ -d $datapath/$dataset ] || mkdir -p $datapath/$dataset
done

# Extract the raw text and meta-information into single files
# Output to data/raw/{train,dev,test}/{train,dev,test}.tsv
echo ">>> Extracting raw text and meta-information from original files."
echo "         Outputting to data/raw/{train,dev,test}/{train,dev,test}.tsv"
python $thisdir/create_dataset_splits.py $raw_folder/TableOfContent.tsv $raw_folder/split $datapath/

# Create individual files for src, trg and meta info for each set
# Output to data/raw/{train,dev,test}/{train,dev,test}.{src,trg,meta}
echo ">>> Creating individual src, trg and meta files for each set."
echo "         Outputting to data/raw/{train,dev,test}/{train,dev,test}.{src,trg,meta}"
bash $thisdir/tsv_to_separate_files.sh $datapath/

# Remove any identical test sentences from the train and dev sets (either side of the data)
# This can happen for very common sentences that may appear in several different texts
# Also deduplicate train set
echo ">>> Removing duplicates from train set (those that appear in test set) of length >= 4 tokens."
echo "         Outputting to $datapath/train/train.filt.tsv"
echo ">>> Also deduplicating the train and dev sets"
python $thisdir/remove_duplicates.py $datapath/test/test $datapath/train/train -t 4 | \
    LC_ALL=C sort -V -u > $datapath/train/train.filt.dedup.tsv
echo ">>> Removing duplicates from dev set (those that appear in test set) of length >= 4 tokens."
echo "         Outputting to $datapath/train/dev.filt.tsv"
python $thisdir/remove_duplicates.py $datapath/test/test $datapath/dev/dev -t 4 | \
    LC_ALL=C sort -V -u > $datapath/dev/dev.filt.dedup.tsv
echo ">>> Separating out filtered tsv files to separate src and trg files"
cat $datapath/train/train.filt.dedup.tsv | cut -f1 | perl -pe 's/^ *//g; s/ *$//g' > $datapath/train/train.filt.dedup.src
cat $datapath/train/train.filt.dedup.tsv | cut -f2 | perl -pe 's/^ *//g; s/ *$//g' > $datapath/train/train.filt.dedup.trg
cat $datapath/train/train.filt.dedup.tsv | cut -f3- | perl -pe 's/^ *//g; s/ *$//g' > $datapath/train/train.filt.dedup.norm.meta
cat $datapath/dev/dev.filt.dedup.tsv | cut -f1 | perl -pe 's/^ *//g; s/ *$//g' > $datapath/dev/dev.filt.dedup.src
cat $datapath/dev/dev.filt.dedup.tsv | cut -f2 | perl -pe 's/^ *//g; s/ *$//g' > $datapath/dev/dev.filt.dedup.trg
cat $datapath/dev/dev.filt.dedup.tsv | cut -f3- | perl -pe 's/^ *//g; s/ *$//g' > $datapath/dev/dev.filt.dedup.norm.meta
cp $datapath/test/test.meta $datapath/test/test.norm.meta

# Pre-normalisation
# (i.e. homogenise certain typographic variants e.g. apostrophes, quotes, ligatures, double consonants, etc.)
echo ">>> Pre-normalise certain typograohic variants (quote marks and apostrophes)"
for lang in src trg; do
    for dataset in train dev; do
	cat $datapath/$dataset/$dataset.filt.dedup.$lang | \
	    bash $thisdir/../norm-scripts/pre-normalise.sh > \
		 $datapath/$dataset/$dataset.filt.dedup.norm.$lang
    done
    cat $datapath/test/test.$lang | bash $thisdir/../norm-scripts/pre-normalise.sh > $datapath/test/test.norm.$lang
done

echo ">>> Setting links to final versions of the raw datasets. "
echo "         Each is accessible via data/raw/{train,dev,test}/{train,dev,test}.finalised.tsv"
for lang in src trg meta; do
    ln -f -s $datapath/train/train.filt.dedup.norm.$lang $datapath/train/train.finalised.$lang
    ln -f -s $datapath/dev/dev.filt.dedup.norm.$lang $datapath/dev/dev.finalised.$lang
    ln -f -s $datapath/test/test.norm.$lang $datapath/test/test.finalised.$lang
done

echo ">>> Separating dev and test sets into separate components"
bash $thisdir/separate_sets_by_genre.sh $datapath/
