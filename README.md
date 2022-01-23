# ModFr-Normalisation


## Citation

## Requirements

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

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

Below you can find normalisation commands for each of the methods compared. All methods take a text from standard input and output normalised text to standard output. Here, the dev (`data/raw/dev/dev.finalised.src`) is used as an example.

**Rule-based:**

```
cat data/raw/dev/dev.finalised.src | bash norm-scripts/rule-based.sh > outputs/rule-based/dev-1.trg
```


**ABA, alignment-based approach:**
```
TODO
```

**Statistical MT (SMT):**

```
>> bash norm-scripts/smt_translate.sh <model_folder>
```
E.g.
```
>> cat data/raw/dev.finalised.src bash norm-scripts/smt_translate.sh mt-models/smt/1/model > outputs/smt/dev/dev-1.trg
```
N.B. If you want to use this script to translate SMT models that have been trained with other segmentations, make sure to change `segtype` in `smt_translate.sh`.


**Neural MT (NMT), both LSTM and Transformer**

```
>> bash norm-scripts/nmt_translate.sh <model_path>
```
E.g.
```
>> cat data/raw/dev.finalised.src | bash norm-scripts/nmt_translate.sh mt-models/lstm/1/checkpointTODO > outputs/lstm/dev/dev-1.trg
>> cat data/raw/dev.finalised.src | bash norm-scripts/nmt_translate.sh mt-models/transformer/1/checkpointTODO > outputs/transformer/dev/dev-1.trg
```

**Post-processing using the contemporary French lexicon, the Le*fff* (Sagot, 2009):**

This approach can be applied after any of the other approaches.

```
cat outputs/rule-based/dev-1.pred.trg | bash norm-scripts/lex-postproc.sh > cat outputs/rule-based+lex/dev-1.pred.trg
```
or applied directly after the main approach:
```
cat data/raw/dev/dev.finalised.src | bash norm-scripts/rule-based.sh | bash norm-scripts/lex-postproc.sh > outputs/rule-based+lex/dev-1.pred.trg
```

## Evaluation

### Evaluate with individual metrics

```
>> bash eval-scripts/bleu.sh <ref_file> <pred_file> fr
>> bash eval-scripts/chrf.sh <ref_file> <pred_file> fr
>> python eval-scripts/levenshtein.py <ref_file> <pred_file> -a {ref,pred} (-c <cache_file>)
>> python eval-scripts/word_acc.py <ref_file> <pred_file> -a {ref,pred,both} (-c <cache_file>)
```
where `-a ref` means that the reference is used as basis for the alignment, `-a pred` that the prediction is used as basis for the alignment, and `-a both` that the average of the two is calculated. An optional cache file destination (format .pickle) can be specified to speed up evaluation when running it several times.

To calculate the average of a metric over several outputs (relevant for different random seeds of the MT approaches):

### Evaluation over multiple metrics

```
>> bash eval-scripts/eval_detailed.sh <output_folder> <ref_file> (<cache_file>)
```
where `output_folder` is the folder containing prediction files to be included in the evaluation (all files ending in `.trg` will be included for evaluation. E.g.

```
>> bash eval-scripts/eval_all.sh outputs/rule-based/dev data/raw/dev/dev.finalised.trg outputs/.cache.pickle 
>> 89.50 & 89.60 & 0.00 & 74.26 & 0.91 \\
```

### Detailed evaluation (including on data subsets)

To calculate all evaluation scores, including on subsets of the data (as specified above and in the meta data):
```
>> bash eval-scripts/eval-all.sh <ref_file> <meta_file> <pred_file> (<cache_file>)
```
E.g.
```
>> bash eval-scripts/eval-all.sh data/raw/dev/dev.finalised.trg data/raw/dev/dev.finalised.meta outputs/rule-based/dev-1.pred.trg outputs/.cache.pickle
>> all,bleu=74.2593 all,chrf=0.90544 all,lev_char=0.0 all,wordacc_r2h0.896 all,wordacc_h2r=0.894 all,wordacc_sym=0.895 1-standard,bleu=73.3256 1-standard,chrf=0.90136 1-standard,lev_char=0.0 1-standard,wordacc_r2h0.892 1-standard,wordacc_h2r=0.891 1-standard,wordacc_sym=0.892 4-medecine-dev,bleu=83.9066 4-medecine-dev,chrf=0.94225 4-medecine-dev,lev_char=0.0 4-medecine-dev,wordacc_r2h0.933 4-medecine-dev,wordacc_h2r=0.929 4-medecine-dev,wordacc_sym=0.931 5-physique-dev,bleu=73.8570 5-physique-dev,chrf=0.90849 5-physique-dev,lev_char=0.0 5-physique-dev,wordacc_r2h0.901 5-physique-dev,wordacc_h2r=0.895 5-physique-dev,wordacc_sym=0.898
```
where `r2h` means that the reference is used as basis for the alignment, `h2r` that the hypothesis is used as basis for the alignment and `sym` means that the mean of the two directions is calculated.

## Retrain a model:

### Preprocessing and binarisation

To preprocess with all segmentations used in our experiments, run the following script:

```
bash data-scripts/process_for_mt.sh
```

This involves:

- preparation of data (+ meta information (decades and years))
- subword segmentation using sentencepiece for the following (joint) vocab sizes:
  - char, 500, 1k, 2k, 4k, 8k, 16k, 24k
- binarisation of the data in the fairseq format (for neural models)

### Hyper-parameter searches

Create model folders and scripts for different hyper-parameter settings as follows:
```
bash mt-training-scripts/create_experiments.sh
```
N.B. You can change the hyper-parameter values in this file to generate different combinations. The dropout, batch size and learning rate are hard-coded as we only try different combinations for a few experiments.

This script will create a model folder named with the specific parameters. Each folder will have a subfolder indicating the random seed and in each of these folders will be the training script. E.g. `mt-models/transformer_char_2enc_2dec_2heads_128embdim_512ff_0.3drop_0.001lr_3000bsz/{1,2,3}/`

To run training:
```
cd MODEL_FOLDER/SEED
bash train.sh
```
Then translate the validation (dev) set for each of the model checkpoints:
```
cd MODEL_FOLDER/SEED
bash translate_val.sh
```
To choose the best checkpoint (using as the criterion symmetrised word accuracy):
```
mt-training-scripts/eval_val.sh MODEL_FOLDER/SEED
```
This will produce a validation file `valid.eval` in the subfolder, which records the scores for each of the checkpoints, finds the best scoring checkpoint and copies it over to `checkpoint_bestwordacc_sym.pt`. The translation of the validation set by this best checkpoint is `checkpoint_bestwordacc_sym.pt.valid.postproc`.


