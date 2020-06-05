import json
from kaggle.api.kaggle_api_extended import KaggleApi

# build code file for Kaggle kernel from options passed w CLI
def build_script_from_options(data, lang, emb, max_epoch, storage) -> None:

    """
    >>> build_script_from_options("conll2001", "nl", ['ft', 'ohe'], 100, 'cpu')
    True
    """
    source = []

    with open("run_NER_experiment.py", "r") as f:
        source = [line for line in f]

    # TODO: options -> txt file -> build_from_file
    with open("run_NER_experiment.py", "w") as f:

        source[
            :10
        ] = f"""
############################
# Experiment configuration #
############################
DATA = '{data}'
LANG = '{lang}'
EMB = {emb}
MAX_EPOCH = {max_epoch}
STORAGE = '{storage}'
############################
            """

        for i in range(len(source)):
            f.write(source[i])


def build_script_from_file(file):
    # TODO: batch init from folder of config files
    pass


class KagglePusher:
    def __init__(self, data, lang, emb, storage):

        self.kernel_name = f"{data}_{lang}_{'_'.join(emb)}"
        self.data = data
        # script will be run on gpu if the storage_mode is set to gpu
        # can be adjusted manually when the embeddings don't fit in memory
        self.gpu = "true" if storage == "gpu" else "false"

    def init_metadata(self) -> None:

        metadata = {
            "id": f"achtuur/{self.kernel_name}",
            "title": self.kernel_name,
            "code_file": "run_NER_experiment.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": self.gpu,
            "enable_internet": "true",
            "dataset_sources": [f"achtuur/{self.data}"],
            "competition_sources": [],
            "kernel_sources": [],
        }

        with open("kernel-metadata.json", "w") as f:
            json.dump(metadata, f)

    def push(self) -> None:

        # push to Kaggle
        api = KaggleApi()
        api.authenticate()
        api.kernels_push("./")
