# ModFr-Normalisation


## Download and prepare data

Get dataset splits:
```
bash data-scripts/get_datasets.sh
```


_What does this do?_

- splitting the files into train/dev/test
- filtering out sentences from the train and dev sets that also appear in the test set and contain over 4 tokens
- normalisation of quotes, apostrophes and repeated spaces

The final raw files are found in `data/raw/{train,dev,test}/{train,dev,test}.finalised.{src,trg,meta}`.

_Subsets of dev and test_

Subsets of the dev and test sets are also available in the same subfolders. 
They represent different scenarios that could be used for separate analysis.

- 1-standard:
    - "belles-lettres" sentences taken from the same distribution as train (80%/10%/10% train/dev/test)
- 2-test: "zero-shot" sentences
    - selected texts, distributed across periods and genres (0%/0%/100% train/dev/test)
- 3-test+train: "few-shot" sentences
    - selected texts, distributed across periods and genres (10%/0%/90% train/dev/test)
- 4-medecine: medical domain
    - 1 document in the dev and the other in the test (two very different documents) (none in train)
- 5-physique: physics/mechanics domain
    - 1 document in dev and the other in test (none in train)


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