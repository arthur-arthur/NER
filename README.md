# NER benchmarking: monolingual vs. multilingual embeddings

This repository bundles different NER benchmark experiments to compare the performance of monolingual and multilingual embeddings for both monolingual (Dutch, French and English) and multilingual datasets. For all experiments, the Flair library was used.  The experiments were run using Kaggle scripts. A simple CLI was used to generate a script and metadata file to push the scripts to Kaggle using the [Kaggle API](https://github.com/Kaggle/kaggle-api).

## Datasets

| Language     	| Dataset   	| Downsampled 	| Tokens (train) 	| Tokens (dev) 	| Tokens (test) 	|
|--------------	|-----------	|-------------	|----------------	|--------------	|---------------	|
| Dutch        	| CoNLL2002 	| No          	|                	|              	|               	|
| French       	| WikiNER   	| Yes (0.10)  	|                	|              	|               	|
| English      	| CoNLL2003 	| No          	|                	|              	|               	|
| Multilingual 	| All       	| No          	|                	|              	|               	|

Note: all datasets were converted to standard CoNLL (BIO)-format. Sentences containing more than 250 tokens (i.e. a total of 3 sentences in the CoNLL2002 dataset) were removed to allow the use of BERT embeddings (limited input sequence length). All three datasets were combined to obtain a multilingual dataset.