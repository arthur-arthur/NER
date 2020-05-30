#########################
# Experiment Config
#########################
DATA = "conll2003"
LANG = "en"  # !important: determines the correct transformer and bytepair embedding
EMB = ["flair", "bpe"]
MAX_EPOCHS = 100
STORAGE = "cpu"
#########################

import json
from kaggle.api.kaggle_api_extended import KaggleApi

# change config lines in NER experiment script
with open("kaggle_pusher.py", "r") as f:
    config = f.readlines()[:8]
source = []

with open("NER_experiment_runner.py", "r") as f:
    source = [line for line in f]

with open("NER_experiment_runner.py", "w") as f:
    source[:8] = config
    for i in range(len(source)):
        f.write(source[i])

# create metadata file
dataset_name = f"{DATA}_{LANG}_{'_'.join(EMB)}"

# script will be run on gpu if the storage_mode is set to gpu
# adjust manually when the embeddings don't fit in memory
enable_gpu = "true" if STORAGE == "gpu" else "false"

metadata = {
    "id": f"achtuur/{dataset_name}",
    "title": dataset_name,
    "code_file": "NER_experiment_runner.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": enable_gpu,
    "enable_internet": "true",
    "dataset_sources": [f"achtuur/{DATA}"],
    "competition_sources": [],
    "kernel_sources": [],
}

with open("kernel-metadata.json", "w") as f:
    json.dump(metadata, f)

# push to Kaggle
api = KaggleApi()
api.authenticate()
api.kernels_push("./")
