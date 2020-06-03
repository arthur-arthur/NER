############################
# Experiment configuration #
############################
DATA = 'nl'                           # [nl, fr, bi]
LANG = 'multi'                        # embedding language [nl, fr, multi]
EMB = ('bert', 'bpe', 'char', 'ohe')
SEEDS = (1, 2, 3)
MAX_EPOCH = 100
STORAGE = 'cpu'
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

# helper functions for data pp: BIO-format, remove punctuation-only tokens and random 60/20/20 split

# helper functions
def is_only_punct(token):
    return all(t in string.punctuation for t in token)


def str_to_BIO(txt, ignore_punct=True):
    return "".join(
        t + " O\n" for t in txt.split()
        if not all([is_only_punct(t), ignore_punct])
    )


def ent_to_BIO(txt, entity, ignore_punct=True):
    out = ""
    txt = [
        t for t in txt.split()
        if not all([is_only_punct(t), ignore_punct])
    ]

    for i, t in enumerate(txt):
        prefix = "I" if i else "B"
        out += f"{t} {prefix}-{entity}\n"

    return out


def split_dataset(data, seed=1, train_dev=(0.6, 0.2)):

    random.seed(seed)
    n = len(data)
    train, dev, test = np.split(
        random.sample(range(n), k=n),
        [int(train_dev[0] * n), int(1 - train_dev[-1] * n)]
    )

    return [
        ("train", [data[i] for i in train]),
        ("dev", [data[i] for i in dev]),
        ("test", [data[i] for i in test])
    ]


def faktion_to_BIO(
        path: str,
        dir_out: str,
        ignore_punct: bool = False,
        seed: int = 1
) -> None:

    data_str = open(path).read()
    data = json.loads(data_str)

    for filename, split in split_dataset(data, seed=seed):

        # output path for preprocessed train/dev/test dataset
        folder = f"/kaggle/working/{dir_out}/"
        os.makedirs(folder, exist_ok=True)

        with open(f"{folder}{filename}.txt", "w") as f:

            for sentence in split:

                entities = [
                    (e['start'], e['end'], e['entity'])
                    for e in sentence['entities']
                ]

                txt, lag = sentence['text'], 0

                for start, stop, entity in sorted(entities, key=lambda x: x[0]):
                    print(str_to_BIO(txt[lag: start], ignore_punct=ignore_punct), file=f, end="")
                    print(ent_to_BIO(txt[start: stop], entity, ignore_punct=ignore_punct), file=f, end="")
                    lag = stop

                # add last part
                print(str_to_BIO(txt[lag:], ignore_punct=ignore_punct), file=f)


def extract_metrics(r):
    """
    input detailed_results attribute of Results class instance
    return list of weighted (micro) avg:
        [precision, recall, accuracy, f1]
    """

    r_lst = r.strip().split("\n")
    metrics = {m: [] for m in ["precision", "recall", "accuracy", "f1"]}
    weights = []

    for d in r_lst[-(len(r_lst) - 2):]:  # substract micro macr avg lines

        label, *rest = d.split()
        m = [float(rest[i]) for i in range(1, 23, 3)]
        weights.append(np.sum(m[:4]))  # first 4 elements: tp, fp, fn, tn,
        for val, k in zip(m[4:], metrics):  # elements 5:9 = pr, rec, acc, f1
            metrics[k].append(val)

    return [
        np.average(v, weights=weights)
        for v in metrics.values()
    ]



class NER_experiment:
    def __init__(self, dataset_name, output_path="/kaggle/working"):

        # input path preprocessed datasets is Kaggle wd
        input_path = f"/kaggle/working/{dataset_name}"

        self.tag_type = "ner"
        self.corpus = ColumnCorpus(
            data_folder=input_path, column_format={0: "text", 1: "pos", 2: "ner"}
        )

        self.tag_dictionary = self.corpus.make_tag_dictionary(tag_type=self.tag_type)

        self.dataset_name = "faktion_" + dataset_name
        self.output_path = output_path

    def build_embedding(self, lang, embedding_codes: List[str]) -> None:

        """
        pass a list of abbreviated embedding codes as the embedding_codes argument.
        If appropriate, the different embedding classes are instantiated according to the lang argument
        Abbr:
            "bert": BERTje (nl), CamemBERT (fr), BERT (en), mBERT (multi) (base cased)
                    (always last layer, first subtoken embedding, ie no scalar mix)
            "bpe": BytePairEmbeddings (fr, nl, en or multi)
            "ohe": OneHotEmbeddings
            "char": CharacterEmbeddings
            "ft": fastText nl/fr/en (WordEmbeddings)
            "flair": flair nl/fr/en, both fw and bw
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

    def collect_results(self, seed):

        # plot training history
        self.history = pd.read_csv(f"{self.output_path}/loss.tsv", sep="\t")
        self._plot_history()

        out = [
            self.dataset_name,          # nl/fr/bi
            self.embedding_name,        # eg bert_flair
            self.lang,                  # eg multi (embedding lang)
            seed
        ]
        # add results (pr, re, acc, f1)
        out.extend([round(x, 4) for x in self._extract_from_log().values()])

        return out

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
        "nl": "bert-base-dutch-cased",
        "en": "bert-base-cased",
        "multi": "bert-base-multilingual-cased",
    }


# Initiate DF to collect results
results = pd.DataFrame(
    index=np.arange(500),
    columns=[
        "dataset", "embedding", "emb_lang", "seed",
        "precision", "recall", "accuracy", "f1"
    ]
)

i = 0

for seed in SEEDS:

    # 1. Convert to BIO format and perform 60/20/20 train/dev/test split
    raw_input_path = f"/kaggle/input/thesis/faktion_training_data_{DATA}-{DATA}.json"
    faktion_to_BIO(raw_input_path, dir_out=DATA, ignore_punct=True, seed=seed)

    # 2. Run experiment on preprocessed DS for given conditions in script header
    experiment = NER_experiment(dataset_name=DATA)
    experiment.build_embedding(LANG, EMB)
    experiment.train(MAX_EPOCH, STORAGE)

    # 3. Store results of experiment in df
    out = experiment.collect_results(seed)

    for col, val in zip(results.columns, out):
        results[col][i] = val

    i += 1