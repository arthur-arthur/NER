# NER benchmarking: monolingual vs. multilingual embeddings

This repository bundles different NER benchmark experiments to compare the performance of monolingual and multilingual embeddings for both monolingual (Dutch, French and English) and multilingual datasets. All experiments made use of the [Flair library](https://github.com/flairNLP/flair). The experiments were run using in individual scripts on Kaggle. A simple CLI was used to generate the script and metadata file to push the scripts to Kaggle using the [Kaggle API](https://github.com/Kaggle/kaggle-api).

## Datasets

| Language     	| Dataset   	| Downsampled 	| Tokens (train) 	| Tokens (dev) 	| Tokens (test) 	|
|--------------	|-----------	|-------------	|----------------	|--------------	|---------------	|
| Dutch        	| CoNLL2002 	| No          	|                	|              	|               	|
| French       	| WikiNER   	| Yes (0.10)  	|                	|              	|               	|
| English      	| CoNLL2003 	| No          	|                	|              	|               	|
| Multilingual 	| All       	| No          	|                	|              	|               	|

Note: all datasets were converted to standard CoNLL (BIO)-format. Sentences containing more than 250 tokens (i.e. a total of 3 sentences in the CoNLL2002 dataset) were removed to allow the use of BERT embeddings (limited input sequence length). All three datasets were combined to obtain a multilingual dataset.

## Embeddings

Different monolingual and multilingual embeddings were tested. This included 2 contextualized embeddings (language-specific or multilingual BERT and Flair) stacked with trainable embeddings (`OneHotEmbeddings` and `CharacterEmbeddings` Lample et al. (2016)) and fixed non-contextualized embeddings (fastText, BytePair embeddings). In addition, stackings of the contextualized embeddings (BERT and Flair) were tested. A full overview of the embeddings that were used is given below:

| Name                  	| Language     	| Type                           	| Ref                                                                                         	|
|-----------------------	|--------------	|--------------------------------	|---------------------------------------------------------------------------------------------	|
|  BERT                 	| English      	| bert-base-cased                	|                                                                                             	|
| BERTje                	| Dutch        	| bert-dutch-base-cased          	| https://github.com/wietsedv/bertje                                                          	|
| CamemBERT             	| French       	| camembert-base                 	| https://camembert-model.fr/                                                                 	|
| multilingual BERT     	| multilingual 	| bert-base-multilingual-cased   	|                                                                                             	|
| Flair                 	| English      	| en-forward + en-backward       	| https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md 	|
| Flair NL              	| Dutch        	| nl-forward + nl-backward       	| https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md 	|
| Flair FR              	| French       	| fr-forward + fr-backward       	| https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md 	|
| multilingual Flair    	| multilingual 	| multi-forward + multi-backward 	|                                                                                             	|
| fastText            	| English      	|                                	|                                                                                             	|
| fastText            	| Dutch        	|                                	|                                                                                             	|
| fastText            	| French       	|                                	|                                                                                             	|
| BytePair            	| English     	|                                	|                                                                                             	|
| BytePair            	| Dutch        	|                                	|                                                                                             	|
| BytePair            	|              	|                                	|                                                                                             	|
| multilingual BytePair 	|              	|                                	|                                                                                             	|
| Character embeddings  	| NA           	|                                	|                                                                                             	|
| OneHotEmbeddings      	| NA           	|                                	|                                                                                             	|


## Command line interface

Since the Kaggle API does not allow to import utility scripts directly from the command line, every experiment was executed from a self-contained script containing all the code to load the dataset, initialize the embeddings, train and evaluate the model. This script was automatically generated and pushed to Kaggle using a simple command line interface. The following command creates the python script, the `kernel-metadata.json` file (kernel name is the concatenation of dataset, language and the different embeddings) and pushes the script to Kaggle. By default, the embedding storage mode of the ModelTrainer instance is set to 'cpu'. When `--storage` option is set to `gpu`, the GPU is automatically enabled on Kaggle.

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
  --storage TEXT    Embedding storage mode. When set to gpu, gpu will be automatically enabled on Kaggle [default: cpu]
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
