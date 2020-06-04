
############################
# Experiment configuration #
############################
DATA = 'conll2003'
LANG = 'multi'
EMB = ['bert', 'flair', 'ft', 'bpe', 'char', 'ohe']
MAX_EPOCH = 100
STORAGE = 'gpu'
############################
                                                                                                                                                                                                                                                

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

import subprocess

subprocess.run("pip install flair", shell=True, check=True)

from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    WordEmbeddings,
    BytePairEmbeddings,
    CharacterEmbeddings,
    TransformerWordEmbeddings,
    OneHotEmbeddings,
)
from flair.trainers import ModelTrainer

from typing import Dict, List


class NER_experiment:
    def __init__(self, dataset_name, output_path="/kaggle/working"):

        input_path = f"/kaggle/input/{dataset_name}"

        self.tag_type = "ner"
        self.corpus = ColumnCorpus(
            data_folder=input_path, column_format={0: "text", 1: "ner"}
        )

        self.tag_dictionary = self.corpus.make_tag_dictionary(tag_type=self.tag_type)

        self.dataset_name = dataset_name
        self.output_path = output_path

    def build_embedding(self, lang, embedding_codes: List[str]) -> None:

        """
        pass a list of abbreviated embedding codes as the embedding_codes argument (CLI: emb arg)
        instantiation of the correct class depends on the lang argument (CLI: --lang option)
        Abbr:
            "bert": BERTje (nl), CamemBERT (fr), BERT (en), mBERT (multi) (base cased)
                    (always last layer, first subtoken embedding, ie no scalar mix)
            "bpe": BytePairEmbeddings (fr/nl/en/multi)
            "ohe": OneHotEmbeddings
            "char": CharacterEmbeddings
            "ft": fastText nl/fr/en (WordEmbeddings)
            "flair": flair (nl/fr/en/multi) -> fw + bw for all
        """

        self.tic = time.time()
        self.embedding_name: str = "-".join(embedding_codes)
        self.lang = lang

        embedding_types: List[TokenEmbeddings] = []

        for code in embedding_codes:

            code = code.lower()
            assert code in [
                "bpe",
                "bert",
                "flair",
                "ft",
                "char",
                "ohe",
            ], f"{code} - Invalid embedding code"

            if code == "ohe":
                embedding_types.append(OneHotEmbeddings(corpus=self.corpus))
            elif code == "ft":
                embedding_types.append(WordEmbeddings(self.lang))
            elif code == "bpe":
                embedding_types.append(BytePairEmbeddings(self.lang))
            elif code == "bert":
                embedding_types.append(
                    TransformerWordEmbeddings(
                        model=self.huggingface_ref[self.lang],
                        pooling_operation="first",
                        layers="-1",
                        fine_tune=False,
                    )
                )
            elif code == "char":
                embedding_types.append(CharacterEmbeddings())
            elif code == "flair":
                embedding_types.append(FlairEmbeddings(f"{self.lang}-forward"))
                embedding_types.append(FlairEmbeddings(f"{self.lang}-backward"))

        self.embedding: StackedEmbeddings = StackedEmbeddings(
            embeddings=embedding_types
        )

        self.tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=self.embedding,
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            use_crf=True,
        )

        self.trainer: ModelTrainer = ModelTrainer(self.tagger, self.corpus)

    def train(self, max_epochs=100, storage_mode="cpu") -> None:

        self.trainer.train(
            base_path=self.output_path,
            learning_rate=0.1,
            mini_batch_size=32,
            max_epochs=max_epochs,
            embeddings_storage_mode=storage_mode,
        )

    def output_results(self) -> None:

        # plot training history
        self.history = pd.read_csv(f"{self.output_path}/loss.tsv", sep="\t")
        self._plot_history()

        dest = self.output_path + "/results.csv"

        out = [
            self.dataset_name,
            self.embedding_name,
            self.lang,
            self.history.EPOCH.iloc[-1],
            round(time.time() - self.tic),
        ]

        out.extend([round(x, 4) for x in self._extract_from_log().values()])
        out = [str(x) for x in out]

        out_names = [
            "dataset",
            "embedding",
            "lang",
            "epochs",
            "duration",
            "precision",
            "recall",
            "accuracy",
            "f1",
        ]

        # write header to file
        if not os.path.exists(dest):
            with open(dest, "w") as f:
                print(",".join(out_names), file=f)
        # write results to file
        with open(dest, "a") as f:
            print(",".join(out), file=f)

    def _extract_from_log(self) -> Dict[str, int]:

        # TODO: compute from test set
        log: List[str] = open(self.output_path + "/training.log", "r").readlines()[-5:-1]
        out = {m: [] for m in ["precision", "recall", "accuracy", "f1"]}
        weights = []

        for entity_class in log:

            label, *result_str = entity_class.split()
            results = [float(result_str[i]) for i in range(1, 23, 3)]

            weights.append(
                np.sum(results[:4])
            )  # tp, fp, fn, tn  -->  count instances/class

            for value, k in zip(results[4:], out):  # pr, rec, acc, f1
                out[k].append(value)

        # micro-average precision, recall, acc, F1
        return {m: np.average(v, weights=weights) for m, v in out.items()}

    def _plot_history(self) -> None:

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 8))
        self.history.plot(
            "EPOCH", ["TRAIN_LOSS", "DEV_LOSS", "LEARNING_RATE"], ax=axes[0]
        )
        self.history.plot(
            "EPOCH", ["DEV_RECALL", "DEV_PRECISION", "DEV_F1"], ax=axes[1]
        )
        fig.savefig(f"{self.output_path}/{self.dataset_name}_{self.embedding_name}.png")

    # wrapper
    def run(self, lang, emb, max_epochs, storage):

        self.build_embedding(lang, emb)
        self.train(max_epochs, storage)
        self.output_results()

    # https://huggingface.co/models references
    huggingface_ref = {
        "fr": "camembert-base",
        "nl": "wietsedv/bert-base-dutch-cased",
        "en": "bert-base-cased",
        "multi": "bert-base-multilingual-cased",
    }


# Run experiment given conditions in script header
exp = NER_experiment(dataset_name=DATA)
exp.run(LANG, EMB, MAX_EPOCH, STORAGE)
