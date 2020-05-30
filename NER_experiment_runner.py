#########################
# Exp Conf
#########################
DATA = 'wikiner'
LANG = 'fr'
EMB = ['bert', 'ft', 'bpe', 'ohe', 'char']
MAX_EPOCHS = 100
STORAGE = 'gpu'
#########################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import datetime

import subprocess
subprocess.run('pip install flair', shell=True, check=True)
subprocess.run('pip install yagmail', shell=True, check=True)

import yagmail
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.embeddings import (
    TokenEmbeddings, StackedEmbeddings, FlairEmbeddings,
    WordEmbeddings, BytePairEmbeddings, CharacterEmbeddings,
    TransformerWordEmbeddings, OneHotEmbeddings
)
from flair.trainers import ModelTrainer

from typing import Dict, List, Tuple, Union

class NER():

    def __init__(self, input_path, output_path="/kaggle/working"):

        self.tag_type = "ner"
        self.corpus = ColumnCorpus(
            data_folder=input_path,
            column_format={0: 'text', 1: 'pos', 2: 'ner'}
        )

        self.tag_dictionary = self.corpus.make_tag_dictionary(tag_type=self.tag_type)

        *_, self.dataset_name = input_path.split("/")
        self.output_path = output_path

    def build_embedding(self, lang, embedding_codes: List[str]) -> None:

        self.tic = time.time()
        self.embedding_name: str = "-".join(embedding_codes)
        self.lang = lang

        embedding_types: List[TokenEmbeddings] = []

        for code in embedding_codes:

            code = code.lower()
            assert code in ["bpe", "bert", "flair", "ft", "char", "ohe"], f"{code} - Invalid embedding code"

            if code == "ohe":
                embedding_types.append(OneHotEmbeddings(corpus=self.corpus))
            elif code == "ft":
                embedding_types.append(WordEmbeddings(self.lang))
            elif code == "bpe":
                embedding_types.append(BytePairEmbeddings(self.lang))
            elif code == "bert":
                embedding_types.append(TransformerWordEmbeddings(
                    model=self._huggingface_ref[self.lang],
                    pooling_operation="first",
                    layers="-1",
                    fine_tune=False
                ))
            elif code == "char":
                embedding_types.append(CharacterEmbeddings())
            elif code == "flair":
                embedding_types.append(FlairEmbeddings(f"{self.lang}-forward"))
                embedding_types.append(FlairEmbeddings(f"{self.lang}-backward"))

        self.embedding: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        self.tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=self.embedding,
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            use_crf=True
        )

        self.trainer: ModelTrainer = ModelTrainer(self.tagger, self.corpus)

    def train(self, max_epochs=100, storage_mode="cpu"):

        self.trainer.train(
            base_path=self.output_path,
            learning_rate=0.1,
            mini_batch_size=32,
            max_epochs=max_epochs,
            embeddings_storage_mode=storage_mode
        )

    def output_results(self) -> None:

        # plot training history
        self.history = pd.read_csv(f"{self.output_path}/loss.tsv", sep='\t')
        self._plot_history()

        dest = self.output_path + "/results.csv"

        out = [
            self.dataset_name,
            self.embedding_name,
            self.lang,
            self.history.EPOCH.iloc[-1],
            round(time.time() - self.tic)
        ]

        out.extend([round(x, 4) for x in self._extract_from_log().values()])
        out = [str(x) for x in out]

        self.out = out # TODO: check if required for report in e-mail body
        self.out_names = ["dataset", "embedding", "lang", "epochs", "duration", "precision", "recall", "accuracy", "f1"]

        # write header to file
        if not os.path.exists(dest):
            with open(dest, "w") as f:
                print(",".join(self.out_names), file=f)
        # write results to file
        with open(dest, "a") as f:
            print(",".join(out), file=f)

    def _extract_from_log(self) -> Dict[str, int]:

        log: str = open(self.output_path + "/training.log", "r").readlines()[-5:-1]
        out = {m: [] for m in ["precision", "recall", "accuracy", "f1"]}
        weights = []

        for entity_class in log:

            label, *result_str = entity_class.split()
            results = [float(result_str[i]) for i in range(1, 23, 3)]

            weights.append(np.sum(results[:4]))     # tp, fp, fn, tn,
            for value, k in zip(results[4:], out):  # pr, rec, acc, f1
                out[k].append(value)

        # micro-average precision, recall, acc, F1
        return {m: np.average(v, weights=weights) for m, v in out.items()}

    def _plot_history(self) -> None:

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 8))
        self.history.plot("EPOCH", ['TRAIN_LOSS', 'DEV_LOSS', "LEARNING_RATE"], ax=axes[0])
        self.history.plot("EPOCH", ["DEV_RECALL", "DEV_PRECISION", "DEV_F1"], ax=axes[1])
        fig.savefig(f"{self.output_path}/{self.dataset_name}_{self.embedding_name}.png")

    # https://huggingface.co/models references
    _huggingface_ref = {
        "fr": "camembert-base",
        "nl": "bert-base-dutch-cased",
        "en": "bert-base-cased",
        "multi": "bert-base-multilingual-cased"
    }

# run experiment
exp = NER(input_path=f"/kaggle/input/{DATA}")
exp.build_embedding(lang=LANG, embedding_codes=EMB)
exp.train(max_epochs=MAX_EPOCHS, storage_mode=STORAGE)
exp.output_results()

# send report via e-mail
kernel_name = f"{DATA}_{LANG}_{'_'.join(EMB)}"
yag = yagmail.SMTP('vrachtstuur', 'eentweedrie123')

yag.send(
    to="arthurleloup@gmail.com",
    subject=kernel_name + "finished" + 3 * "\U0001F389",
    contents=[
        "-".join(x.ljust(15) for x in exp.out_names),
        "-".join(x.ljust(15) for x in exp.out),
        {'/kaggle/working/history.png': 'training history plot'},
        {'/kaggle/working/results.csv': 'results'}
    ]
)