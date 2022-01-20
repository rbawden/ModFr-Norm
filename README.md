# ModFr-Normalisation


## Download and prepare data

Get dataset splits:
```
bash data-scripts/get_datasets.sh
```
This involves:

- splitting the files into train/dev/test
- filtering out sentences from the train and dev sets that also appear in the test set and contain over 4 tokens
- normalisation of quotes, apostrophes and repeated spaces

The final raw files are found in `data/raw/{train,dev,test}/{train,dev,test}.finalised.{src,trg,meta}`.

To ensure that the dev and test sets contain a variety of different genres, including texts not represented in the train set, we distributed sentences as follows:

- 

Subsets of the dev and test sets are also available in the same subfolders. The subsets are determined by the selection method used to create the dataset splits. They represent different scenarios that could be used for separate analysis (same distribution as the train set, few-shot, zero-shot and specific domains (medicine, physics).

- 1-standard:
    - "belles-lettres" sentences taken from the same distribution as train
    - 80% were in the train and 10% in each of the dev and test sets
- 2-test: "zero-shot" sentences
    - concerns selected texts, distributed across periods and genres
    - 100% of sentences go to the test set (i.e. 0% in train and dev)
- 3-test+train: "few-shot" sentences
    - concerns selected texts, distributed across periods and genres
    - 10% of sentences go to the train set and 90% to the test set (none to the dev)
- 4-medecine: medical domain
    - 1 document in the dev and the other in the test (two very different documents)
    - no such documents in the train
- 5-physique: physics/mechanics domain
    - 1 document in dev and the other in test
    - no such documents in the train


## Approaches compared

### Identity function

### Rule-based baseline

### Post-processing using the contemporary French lexicon, the Le*fff* (Sagot, 2009)

### ABA, alignment-based approach

### Machine Translation (MT) approaches

### Preprocessing and binarisation

To preprocess with all segmentations used in our experiments, run the following script:

```
bash data-scripts/process_for_mt.sh

This involves:

- preparation of data with the addition of meta information (decades and years) on the source/target side of the data
- subword segmentation using sentencepiece for the following (joint) vocab sizes:
  - char, 500, 1k, 2k, 4k, 8k, 16k, 24k
- binarisation of the data in the fairseq format (for neural models)

### Statistical MT (SMT)

TODO

### Neural MT (NMT)





## Neural models


```

### Normalse using pretrained models





## Evaluation

To do the full evaluation (on the whole test set plus on individual genres of text):

`bash scripts/full_eval.sh`



## Hyper-parameter searches

TODO