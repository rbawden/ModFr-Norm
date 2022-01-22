# ModFr-Normalisation


## Download and prepare data

Get dataset splits:
```
bash data-scripts/get_datasets.sh
```

The final raw files are found in `data/raw/{train,dev,test}/{train,dev,test}.finalised.{src,trg,meta}`.

**What does this do?**

- splitting the files into train/dev/test
- filtering out sentences from the train and dev sets that also appear in the test set and contain over 4 tokens
- normalisation of quotes, apostrophes and repeated spaces



**Subsets of dev and test**

Subsets of dev/test are available in the same subfolders (different data selection scenarios that could be used for separate analysis).

- _1-standard_: "belles-lettres" sentences taken from the same distribution as train (80%/10%/10% train/dev/test)
- _2-test_ ("zero-shot"): selected texts, distributed across periods and genres (0%/0%/100% train/dev/test)
- _3-test+train_ ("few-shot"): selected texts, distributed across periods and genres (10%/0%/90% train/dev/test)
- _4-medecine_ (medical domain): 1 document in the dev and the other in the test (two very different documents) (none in train)
- _5-physique_ (physics/mechanics domain): 1 document in dev and the other in test (none in train)


## Normalisation approaches

Below you can find normalisation commands for each of the methods compared. All methods take a text from standard input and output normalised text to standard output.

### Rule-based

This approach applies a series of substitutions through a series of manually defined regular expressions:

```
cat data/raw/dev/dev.finalised.src | bash norm-scripts/rule-based.sh > outputs/rule-based/dev-1.pred.trg
```


### ABA, alignment-based approach



### Machine Translation (MT) approaches


### Statistical MT (SMT)

TODO

### Neural MT (NMT)

Normalise with a pretrained model:
```
TODO
```

### Post-processing using the contemporary French lexicon, the Le*fff* (Sagot, 2009)

This approach can be applied after any of the other approaches. It normalises words that match with words in a contemporary French lexicon (the Le*fff*), where certain differences are neutralised (accents, capital letters, etc.). Here the script is applied to the output of the rule-based approach above.

```
cat outputs/rule-based/dev-1.pred.trg | perl norm-scripts/lex-postproc.sh > cat outputs/rule-based+lex/dev-1.pred.trg
```

#### Retrain a model:

### Preprocessing and binarisation

To preprocess with all segmentations used in our experiments, run the following script:

```
bash data-scripts/process_for_mt.sh
```

This involves:

- preparation of data with the addition of meta information (decades and years) on the source/target side of the data
- subword segmentation using sentencepiece for the following (joint) vocab sizes:
  - char, 500, 1k, 2k, 4k, 8k, 16k, 24k
- binarisation of the data in the fairseq format (for neural models)

```
TODO
```

#### Hyper-parameter search

Create model folders and scripts for different hyper-parameter settings as follows:
```
TODO
```
N.B. You can change the hyper-parameter values in this file to generate different combinations.



### Normalse using pretrained models





## Evaluation

To do the full evaluation (on the whole test set plus on individual genres of text):

`bash scripts/full_eval.sh`



## Hyper-parameter searches

TODO
