# NER benchmarking: monolingual vs. multilingual embeddings

This repository bundles different NER benchmark experiments to compare the performance of monolingual and multilingual embeddings for both monolingual (Dutch, French and English) and multilingual datasets. For all experiments, the excellent [Flair library](https://github.com/flairNLP/flair) was used.  The experiments were run using Kaggle scripts. A simple CLI was used to generate a script and metadata file to push the scripts to Kaggle using the [Kaggle API](https://github.com/Kaggle/kaggle-api).

## Datasets

| Language     	| Dataset   	| Downsampled 	| Tokens (train) 	| Tokens (dev) 	| Tokens (test) 	|
|--------------	|-----------	|-------------	|----------------	|--------------	|---------------	|
| Dutch        	| CoNLL2002 	| No          	|                	|              	|               	|
| French       	| WikiNER   	| Yes (0.10)  	|                	|              	|               	|
| English      	| CoNLL2003 	| No          	|                	|              	|               	|
| Multilingual 	| All       	| No          	|                	|              	|               	|

Note: all datasets were converted to standard CoNLL (BIO)-format. Sentences containing more than 250 tokens (i.e. a total of 3 sentences in the CoNLL2002 dataset) were removed to allow the use of BERT embeddings (limited input sequence length). All three datasets were combined to obtain a multilingual dataset.

## Command line interface

The following command creates a python script to run the experiment with the given configuration, creates a `kernel-metadata.json` file (kernel name is the concatenation of dataset, language and the different embeddings) and pushes the script to Kaggle. By default, the embedding storage mode of the ModelTrainer instance is set to 'cpu'. When `--storage` option is set to `gpu`, the GPU is automatically enabled on Kaggle.

Important: the `--data` option requires a valid name of the dataset on Kaggle (path: `/kaggle/input/<data>`)

```bash
$ python3 push_and_run.py --help
Usage: push_and_run.py [OPTIONS] [EMB]...

  Embedding codes can be passed as arguments, the appropriate classes are
  initiated according to the --lang option. Multiple embeddings are
  concatenated (StackedEmbeddings)

  Abbreviations (not case-sensitive):

  =====================================================================

      "bert":     BERTje (nl), CamemBERT (fr), BERT (en), mBERT (multi)

      "bpe":      BytePairEmbeddings (fr, nl, en or multi)

      "ohe":      OneHotEmbeddings

      "char":     CharacterEmbeddings

      "ft":       fastText nl/fr/en (WordEmbeddings)

      "flair":    flair nl/fr/en, both fw and bw

  =====================================================================

Options:
  --data TEXT       Dataset: [conll2002, conll2003, wikiner, trilingual]
  --lang TEXT       Language of embedding class: [en, nl, fr, multi]
  --epochs INTEGER  Max number of epochs  [default: 100]
  --storage TEXT    Embedding storage mode. When set to gpu, the script will
                    also be run on a gpu  [default: cpu]
  --help            Show this message and exit.
```

Example 1: To run an experiment using CamemBERT + French fastText + OneHotEmbeddings on the WikiNER DS:

```bash
$ python3 push_and_run.py --data wikiner --lang fr bert ft ohe
```

Example 2: To run an experiment using mBERT + mFlair + multilingual BytePair embeddings on the trilingual DS:

```bash
$ python3 push_and_run.py --data trilingual --lang multi bert flair bpe
```