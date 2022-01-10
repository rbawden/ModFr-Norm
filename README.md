# ModFr-Normalisation


## Download and prepare data

### Download data

```
git clone https://github.com/e-ditiones/PARALLEL17.git
```


### Get dataset splits
```
bash data-scripts/prepare_data.sh
```
This involves:

- splitting the files into train/dev/test
- extracting the raw text and meta-information into single files for each test set
- creating individual files for src, trg and meta info for each set.
- filtering out sentences from the train and dev sets that also appear in the test set and contain over 4 tokens
- normalisation of quotes, apostrophes and repeated spaces.

The final raw files are found in data/raw/{train,dev,test}/{train,dev,test}.finalised.{src,trg,meta}.


## Neural models

### Preprocessing and binarisation

To preprocess with all segmentations used in our experiments, you can just run this script
(this also includes the binarisation of the data in the fairseq format (using fairseq-preprocess)):
```
bash data-scripts/prepare_processing.sh
```

### Normalse using pretrained models





## Evaluation

To do the full evaluation (on the whole test set plus on individual genres of text):

`bash scripts/full_eval.sh`



## Hyper-parameter searches

TODO