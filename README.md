# ModFr-Normalisation

This repository contains the scripts and models to reproduce the results of the preprint [Automatic Normalisation of Early Modern French](https://hal.inria.fr/hal-03540226). See below for citation instructions.


## Recommended model for easy use

As well as the models trained in the paper (see below for instructions on how to use and retrain them), we distribute one of our models in an easily useable format, distributed by HuggingFace [here](https://huggingface.co/rbawden/modern_french_normalisation). It is a transformer model (equivalent to the one trained in the paper), ported to HuggingFace, fine-tuned, and also includes more rigorous post-processing (which can be disabled for faster normalisation).

To use the model on the command line:
```
cat INPUT_FILE | python hf-conversion/pipeline.py -k BATCH_SIZE -b BEAM_SIZE > OUTPUT_FILE
```

You can also use the pipeline class python-internally as follows (you need to have the pipeline.py file locally to do this):

```
tokeniser = AutoTokenizer.from_pretrained("rbawden/modern_french_normalisation")
model = AutoModelForSeq2SeqLM.from_pretrained("rbawden/modern_french_normalisation")
norm_pipeline = NormalisationPipeline(model=model,
                                      tokenizer=tokeniser,
                                      batch_size=batch_size,
                                      beam_size=beam_size)
                                              
list_inputs = ["1. QVe cette propoſtion, qu'vn eſpace eſt vuidé, repugne au ſens commun.", Adieu, i'iray chez vous tantoſt vous rendre grace.]
list_outputs = norm_pipeline(list_inputs)
print(list_outputs)

>> ["1. QUe cette propôtion, qu'un espace est vidé, répugne au sens commun.", "Adieu, j'irai chez vous tantôt vous rendre grâce."]
```

## Reproducing the results of the paper and using the normalisation models

## Requirements

- Python3.7 and the requirements specified in `requirements.txt`
- [KenLM](https://github.com/kpu/kenlm) (to train language models for SMT)
- [Moses](https://github.com/moses-smt/mosesdecoder) (for training and decoding with SMT models)

```
python3 -m venv modfr_env
source modfr_env/bin/activate
pip install -r requirements.txt
```


### Download and prepare data

#### Parallel training data
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

#### Monolingual normalised data (used for some of the language models used for SMT)

Get monolingual normalised data:
```
TODO download data
python data-scripts/get_monolingual_normalised.py <txt_folder> <toc_folder>
bash data-scripts/process_monolingual.sh # to be updated
```

### Download the models

```
bash data-scripts/download_models.sh
```

### Normalisation approaches

Below you can find normalisation commands for each of the methods compared. All methods take a text from standard input and output normalised text to standard output. Here, the dev (`data/raw/dev/dev.finalised.src`) is used as an example.

To use MT approaches, you must first download the models to the main directory:
```
cd ModFr-Norm
wget http://almanach.inria.fr/files/modfr_norm/mt-models.tar.gz
tar -xzvf mt-models.tar.gz
```

Find below the different commands used for each of the approaches:
```
# Rule-based
cat data/raw/dev/dev.finalised.src | \
  bash norm-scripts/rule-based.sh \
    > outputs/rule-based/dev-1.trg

# SMT: bash norm-scripts/smt_translate.sh <model_folder>
cat data/raw/dev/dev.finalised.src | \
  bash norm-scripts/smt_translate.sh final-mt-models/smt/1/model \
    > outputs/smt/dev/dev-1.trg

# NMT (LSTM): bash norm-scripts/nmt_translate.sh <model_path>
cat data/raw/dev/dev.finalised.src | \
  bash norm-scripts/nmt_translate.sh final-mt-models/lstm/1/checkpoint_bestwordacc_sym.pt \
  > outputs/lstm/dev/dev-1.trg

# NMT (Transformer): bash norm-scripts/nmt_translate.sh <model_path>
cat data/raw/dev/dev.finalised.src | \
  bash norm-scripts/nmt_translate.sh final-mt-models/transformer/1/checkpoint_bestwordacc_sym.pt \
    > outputs/transformer/dev/dev-1.trg
    
# Post-processing using the contemporary French lexicon, the Le*fff* (Sagot, 2009)
# Can be applied after any of the other approaches
cat outputs/rule-based/dev-1.pred.trg | \
  bash norm-scripts/lex-postproc.sh \
    > cat outputs/rule-based+lex/dev-1.pred.trg
```

For ABA, the alignment-based approach, see the github repository: [https://github.com/johnseazer/aba](https://github.com/johnseazer/aba)).


### Evaluation

#### Evaluate with individual metrics

```
bash eval-scripts/bleu.sh <ref_file> <pred_file> fr
bash eval-scripts/chrf.sh <ref_file> <pred_file> fr
python eval-scripts/levenshtein.py <ref_file> <pred_file> -a {ref,pred} (-c <cache_file>)
python eval-scripts/word_acc.py <ref_file> <pred_file> -a {ref,pred,both} (-c <cache_file>)
python eval-scripts/word_acc_oov.py <ref_file> <pred_file> <trg_train_file> -a ref (-c <cache_file>)
```
where `-a ref` means that the reference is used as basis for the alignment, `-a pred` that the prediction is used as basis for the alignment, and `-a both` that the average of the two is calculated. An optional cache file destination (format .pickle) can be specified to speed up evaluation when running it several times.

To calculate the average of a metric over several outputs (relevant for different random seeds of the MT approaches):

#### Evaluation over multiple metrics

```
bash eval-scripts/eval_all.sh <output_folder> <ref_file> (<cache_file>)
```
where `output_folder` is the folder containing prediction files to be included in the evaluation (all files ending in `.trg` will be included for evaluation. E.g.

```
bash eval-scripts/eval_all.sh outputs/rule-based/dev data/raw/dev/dev.finalised.trg outputs/.cache-dev.pickle 

WordAcc (ref) | WordAcc (sym) | WordAcc OOV (ref) | Levenshtein | BLEU | ChrF
-----
89.80 | 89.83 | 65.48 | 2.88 | 74.26 | 90.54
```

#### Detailed evaluation (including on data subsets)

To calculate all evaluation scores, including on subsets of the data (as specified above and in the meta data):
```
bash eval-scripts/eval_detailed.sh <ref_file> <meta_file> <pred_file> (<cache_file>)
```
E.g.
```
>> bash eval-scripts/eval_detailed.sh data/raw/dev/dev.finalised.trg data/raw/dev/dev.finalised.meta outputs/rule-based/dev/dev-1.trg outputs/.cache-dev.pickle
>> all,bleu=74.2593 all,chrf=90.54 all,lev_char=2.88061409315046 all,wordacc_r2h=89.83100878163974 all,wordacc_h2r=89.76316112406431 all,wordacc_sym=89.79708495285203 1-standard,bleu=73.3256 1-standard,chrf=90.14 1-standard,lev_char=3.003617425214223 1-standard,wordacc_r2h=89.44745130416169 1-standard,wordacc_h2r=89.36994660564454 1-standard,wordacc_sym=89.40869895490312 4-medecine-dev,bleu=83.9066 4-medecine-dev,chrf=94.22 4-medecine-dev,lev_char=1.6903731189445474 4-medecine-dev,wordacc_r2h=93.50119088125213 4-medecine-dev,wordacc_h2r=93.41479972844536 4-medecine-dev,wordacc_sym=93.45799530484874 5-physique-dev,bleu=73.8570 5-physique-dev,chrf=90.85 5-physique-dev,lev_char=2.8046184081231575 5-physique-dev,wordacc_r2h=90.18160047894632 5-physique-dev,wordacc_h2r=90.18710191082803 5-physique-dev,wordacc_sym=90.18435119488717
```
where `r2h` means that the reference is used as basis for the alignment, `h2r` that the hypothesis is used as basis for the alignment and `sym` means that the mean of the two directions is calculated.


### Results

#### Dev set

| Method | WordAcc (ref) | WordAcc (sym) | WordAcc (ref) OOV | Levenshtein | BLEU | ChrF |
| --- | --- | --- | --- | --- | --- | --- |
| Identity | 73.92 | 73.95 | 47.91 | 7.72 | 42.33 | 74.95|
| Identity+lex | 86.75 | 86.78 | 70.32 | 3.57 | 68.08 | 88.01 |
| Rule-based | 89.80 | 89.83 | 65.48 | 2.88 | 74.26 | 90.54 |
| Rule-based+lex | 91.69 | 91.71 | 72.26 | 2.33 | 78.91 | 92.45 |
| ABA | 95.72 | 95.76 | 75.17 | 1.21 | 89.19 | 96.38 |
| ABA+lex | 96.07 | 96.11 | 78.92 | 1.06 | 89.89 | 96.73 |
| SMT | **97.61±0.04** | **97.59±0.04** | 77.65±0.16 | **0.63±0.01** | **93.67±0.10** | **98.08±0.03** |
| SMT+lex | **97.76±0.04** | **97.75±0.04** | **81.24±0.19** | **0.59±0.01** | **94.11±0.10** | **98.23±0.03** |
| LSTM | 97.16±0.10 | 96.97±0.08 | **78.30±0.81** | 1.13±0.09 | 92.98±0.33 | 97.60±0.06 |
| LSTM+lex | 97.30±0.14 | 97.11±0.11 | 81.08±0.09 | 1.10±0.09 | 93.36±0.40 | 97.73±0.08 |
| Transformer | 96.79±0.05 | 96.58±0.07 | 76.78±0.71 | 1.26±0.04 | 92.17±0.06 | 97.27±0.05 |
| Transformer+lex | 96.92±0.09 | 96.70±0.10 | 79.10±0.85 | 1.23±0.05 | 92.51±0.17 | 97.40±0.09 |

#### Test set

| Method | WordAcc (ref) | WordAcc (sym) | WordAcc (ref) OOV | Levenshtein | BLEU | ChrF |
| --- | --- | --- | --- | --- | --- | --- |
| Identity | 72.72 | 72.73 | 43.00 | 8.15 | 40.25 | 73.77 |
| identity+lex | 86.12 | 86.12 | 64.84 | 3.78 | 66.78 | 87.40 |
| Rule-based | 89.06 | 89.05 | 60.22 | 3.08 | 72.47 | 89.94 | 
| Rule-based+lex | 90.85 | 90.85 | 66.51 | 2.56 | 76.90 | 91.70 |
| ABA | 95.13 | 95.14 | 69.50 | 1.35 | 87.70 | 95.84 |
| ABA+lex | 95.44 | 95.44 | 73.54 | 1.25 | 88.37 | 96.13 |
| SMT | **97.12±0.02** | **97.10±0.02** | 75.64±0.18 | **0.76±0.01** | **92.59±0.05** | **97.71±0.01** |
| SMT+lex | 97.26±0.02 | 97.24±0.02 | **78.37±0.20** | 0.73±0.01 | 92.97±0.05 | 97.85±0.01 |
| LSTM | 96.52±0.07 | 96.14±0.08 | 76.69±0.70 | 1.66±0.04 | 91.77±0.21 | 96.85±0.08 |
| LSTM+lex | 96.63±0.08 | 96.25±0.10 | **78.35±0.79** | 1.64±0.05 | 92.07±0.25 | 96.95±0.10 |
| Transformer | 96.27±0.05 | 95.89±0.07 | 75.73±0.38 | 1.81±0.01 | 91.30±0.08 | 96.65±0.05 |
| Transformer+lex | 96.39±0.07 | 96.01±0.09 | **77.51±1.00** | 1.78±0.02 | 91.62±0.14 | 96.76±0.08 |

### Alignment

It can be useful to obtain an alignment between either the source file or the reference file. To do this, we can use a command very similar to the evalution scripts:

```
python eval-scripts/align_levenshtein.py <ref_or_src_file> <pred_file> -a {ref,pred} (-c <cache_file>)
python eval-scripts/align_levenshtein.py  data/raw/dev/dev.finalised.trg outputs/smt+lex/dev/dev-1.trg -a ref -c outputs/.cache.pickle
```

The alignment script relies on a non-destructive tokenisation convention whereby a token boundary is marked by two spaces when the tokens are white-spaced separated in the raw input and by a single space when they are not. This means that the initial text is preserved, despite the tokenisation applied. The chosen tokenisation can be modified (in `eval-scripts/utils.py`). By default, we apply a very simple tokenisation, separating on whitespace and certain punctuation marks.

Here is an example:

If the reference file contains the following (made-up) example sentence:
```
surtout j'ai choisi davantage ses écrits
```
and the predicted file contains the following sentence:
```
sur tout ji choisi d'avantage ses escrits
```
the alignment script will output:
```
surtout||||sur▁▁tout  j'||||j░ ai||||i  choisi  davantage||||d'▁avantage  ses  écrits||||escrits
```
In this output, different cases arise:
- aligned token is identical: the token is writte as it is (e.g. `choisi`);
- aligned token is different: the reference token is written first, followed by the separator `||||` and the corresponding predicted (sub)token(s). Tokenisation mismatches between the reference and prediction are marked on the predicted side as follows:
  - oversplitting: when there is a token boundary on the predicted side that does not correspond to a reference token boundary, it is marked using one or two consecutive symbols `▁`, depending on whether the predicted tokens are white-space separated or not (e.g. (1) `surtout||||sur▁▁tout`, where the two-token predicted sequence ```sur tout``` is aligned with the reference token `surtout`; and (2) `davantage||||d'▁avantage`, where the two-token predicted sequence ```d'avantage```, which is tokenised as `d' avantage`, is aligned with the reference token `davantage`);
  - undersplitting: when there is no token boundary on the predicted side at a place where there is one on the reference side, the subtoken aligned with the first reference token is appended with the symbol `░` (e.g. `j'||||j░ ai||||i` means that the predicted token `ji` is aligned to both reference tokens `j'` and `ai`, the `░` allowing for the correct reconstruction of the single predicted token `ji` from the alignment script output).

This token-level alignment is produced based on a character-level alignment obtained using a dedicated variant of the weighted Levenshtein algorithm, designed to avoid tokenisation and punctuation mismatches unless they are really necessary for a successful alignment:
- by default, the cost of a substitution is 1, whereas the cost of an insertion or a deletion is 0.8;
- the cost of a substitution of a reference white-space character with a non-white-space is prohibitive (1,000,000);
- the cost of a substitution of a reference non-white-space character with a white-space is 30;
- the cost of a substitution involving a punctuation mark (within `,.;-!?'`) is 20;
- the cost of the insertion or deletion of a white-space character is prohibitive;
- the cost of the insertion of a white-space character is 2.


### Retrain the MT models

#### Preprocessing and binarisation

To preprocess with all segmentations used in our experiments, run the following script:

```
bash data-scripts/process_for_mt.sh
```

This involves:

- preparation of data (+ meta information (decades and years))
- subword segmentation using sentencepiece for the following (joint) vocab sizes:
  - char, 500, 1k, 2k, 4k, 8k, 16k, 24k
- binarisation of the data in the fairseq format (for neural models)


#### Retraining SMT models

##### Training a language model with KenLM

Train all \textit{n}-gram language model combinations as follows:
```
bash mt-training-scripts/train_lms.sh
```
Make sure to change the tool paths in this file first to point to your installation of [KenLM](https://github.com/kpu/kenlm).

##### Training an SMT model with Moses

An example of a training script is giving in `mt-models/best-smt/1/`. 

To train a new phrase table:
- Create a new model folder (e.g. `mt-models/smt-bpe_joint_1000/1`)
- Copy the train script over: `cp mt-models/best-smt/1/train.sh mt-models/smt-bpe_joint_1000/1/`
- Modify the location of you tools directory `tools=~tools`
- Modify `type=bpe_joint_500` to your chosen segmentation type (e.g. type=bpe_joint_1000)
- Modify the final lines of the two `train-model.perl` commands if you wish to change the type of language model used
- Go to the directory and run training: `cd mt-models/smt-bpe_joint_1000/1; bash train.sh`

Tune the models:
- Copy the tuning script over `cp mt-models/best-smt/1/tune.sh mt-models/smt-bpe_joint_1000/1/`
- As before, modify the tools location and the segmentation type.
- Go to the directory adn run tuning: `cd mt-models/smt-bpe_joint_1000/1; bash tune.sh`

This does tuning for 1 random seed. To do the other two random seeds create two more subfolders `mt-models/smt-bpe_joint_1000/2` and `mt-models/smt-bpe_joint_1000/3`, copy the tuning script over from `1/` as it is and rerun tuning (i.e. you do not need to retrain phrase tables and language models).

#### Hyper-parameter searches for LSTM and Transformer models

Create model folders and scripts for different hyper-parameter settings as follows:
```
bash mt-training-scripts/create_experiments.sh
```
N.B. You can change the hyper-parameter values in this file to generate different combinations. The dropout, batch size and learning rate are hard-coded as we only try different combinations for a few experiments.

This script will create a model folder named with the specific parameters. Each folder will have a subfolder indicating the random seed and in each of these folders will be the training script. 
E.g. `mt-models/transformer_char_2enc_2dec_2heads_128embdim_512ff_0.3drop_0.001lr_3000bsz/{1,2,3}/`

To run training:
```
cd <model_folder>/<seed>
bash train.sh
```
Then translate the validation (dev) set for each of the model checkpoints:
```
cd <model_folder>/<seed>
bash translate_val.sh
```
To choose the best checkpoint (using as the criterion symmetrised word accuracy):
```
bash mt-training-scripts/eval_val.sh <model_folder>/<seed>
```
This will produce a validation file `valid.eval` in the subfolder, which records the scores for each of the checkpoints, finds the best scoring checkpoint and copies it over to `checkpoint_bestwordacc_sym.pt`. The translation of the validation set by this best checkpoint is `checkpoint_bestwordacc_sym.pt.valid.postproc`.


### Citation

If you use or refer to this work, please cite the following paper:

Rachel Bawden, Jonathan Poinhos, Eleni Kogkitsidou, Philippe Gambette, Benoît Sagot, et al. Automatic Normalisation of Early Modern French. 2022. Preprint.

Bibtex:
```
@misc{bawden:hal-03540226,
  title = {{Automatic Normalisation of Early Modern French}},
  author = {Bawden, Rachel and Poinhos, Jonathan and Kogkitsidou, Eleni and Gambette, Philippe and Sagot, Beno{\^i}t and Gabay, Simon},
  url = {https://hal.inria.fr/hal-03540226},
  note = {working paper or preprint},
  year = {2022},
  HAL_ID = {hal-03540226},
  HAL_VERSION = {v1},
}
```

The models can be found on Zenodo:

```
@software{rachel_bawden_2022_6482368,
  author       = {Rachel Bawden},
  title        = {{FreEM-corpora/FreEM-norm-model-smt: SMT 
                   normalisation model for Early Modern French}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6482368},
  url          = {https://doi.org/10.5281/zenodo.6482368}
}

@software{rachel_bawden_2022_6481539,
  author       = {Rachel Bawden},
  title        = {{FreEM-corpora/FreEM-norm-model-LSTM: LSTM 
                   normalisation model for Early Modern French}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6481539},
  url          = {https://doi.org/10.5281/zenodo.6481539}
}

@software{rachel_bawden_2022_6482342,
  author       = {Rachel Bawden},
  title        = {{FreEM-corpora/FreEM-norm-model-transformer: 
                   Transformer normalisation model for Early Modern
                   French}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6482342},
  url          = {https://doi.org/10.5281/zenodo.6482342}
}
```

And to reference the FreEM-norm and FreEM-max datasets used in the experiments:

For FreEM-norm (used to train ABA, SMT and neural models)
Simon Gabay. (2022). FreEM-corpora/FreEMnorm: FreEM norm Parallel corpus (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5865428
```
@software{simon_gabay_2022_5865428,
  author       = {Simon Gabay},
  title        = {{FreEM-corpora/FreEMnorm: FreEM norm Parallel 
                   corpus}},
  month        = jan,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.5865428},
  url          = {https://doi.org/10.5281/zenodo.5865428}
}
```
For FreEM-max (used to train the large language models for SMT):

```
@software{gabay_simon_2022_6481135,
  author       = {Gabay, Simon and
                  Bartz, Alexandre and
                  Gambette, Philippe and
                  Chagué, Alix},
  title        = {{FreEM-corpora/FreEMmax\_OA: FreEM max OA: A Large 
                   Corpus for Early modern French - Open access
                   version}},
  month        = apr,
  year         = 2022,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6481135},
  url          = {https://doi.org/10.5281/zenodo.6481135}
}
```
