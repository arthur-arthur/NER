import flair.datasets
import os
from urllib.request import urlopen

# WikiNER
corpus = flair.datasets.WIKINER_FRENCH().downsample(0.10)


def wikiNER_to_CoNLL(partition, file_out):

    with open(f"/kaggle/working/{file_out}", "w") as f:

        for sentence in partition:

            for token in sentence:

                word = token.text
                tag = token.get_tag("ner").value
                # to CoNLL format (BIO2)
                if tag.startswith("S-"):
                    tag = tag.replace("S-", "B-")
                if tag.startswith("E-"):
                    tag = tag.replace("E-", "I-")

                print(f"{word} {tag}", file=f)

            print("", file=f)


os.makedirs("/kaggle/working/WikiNER", exist_ok=True)

for f, part in zip(
    ["WikiNER/train.txt", "WikiNER/dev.txt", "WikiNER/test.txt"],
    [corpus.train, corpus.dev, corpus.test],
):
    wikiNER_to_CoNLL(part, f)

# CoNLL 2002
def pp_conll2002(split, max_sentence_length=None):

    partitions = {"train": "train", "testa": "dev", "testb": "test"}
    url = f"https://www.clips.uantwerpen.be/conll2002/ner/data/ned.{split}"
    file = open(f"conll2002/{partitions[split]}.txt", "w")

    sentence = []
    c, remove = 0, 0

    with file as f:

        for line in urlopen(url):
            line = line.decode("latin-1").strip("\n")

            if line.startswith("-DOCSTART-"):
                continue

            if line:
                token, pos, ner = line.split()
                sentence.append(f"{token} {ner}")

            else:
                if sentence:
                    if max_sentence_length is None:
                        print("\n".join(sentence), file=f)
                        c += 1

                    elif len(sentence) < max_sentence_length:
                        print("\n".join(sentence), file=f)
                        c += 1

                    else:
                        remove += 1

                    print("", file=f)
                    sentence = []

    return c, remove


for split in ["train", "testa", "testb"]:

    retain, remove = pp_conll2002(split, max_sentence_length=250)
    print(f"{split.ljust(10, '-')}>Removed {remove}/{retain} sentences")


# CoNLL2003
def pp_conll2003(path_in, path_out):

    sentence = []
    in_entity = (
        False  # track if currently in multi-token entity (BIO -> BIO2 conversion)
    )

    with open(path_out, "w") as o:

        with open(path_in, "r") as i:

            for line in i:
                line = line.strip("\n")

                if line.startswith("-DOCSTART-"):
                    continue

                if line:
                    token, *_, ner = line.split()
                    if ner == "O":
                        in_entity = False
                    elif ner.startswith("I-") and not in_entity:
                        ner = ner.replace("I-", "B-")
                        in_entity = True
                    elif ner.startswith("B-"):
                        in_entity = True

                    sentence.append(f"{token} {ner}")

                else:
                    if sentence:
                        print("\n".join(sentence), file=o)
                        print("", file=o)

                    sentence = []


for file_in, file_out in zip(
    ["eng.train", "eng.testa", "eng.testb"], ["train.txt", "dev.txt", "test.txt"]
):
    path_in = f"conll2003/raw/{file_in}"
    path_out = f"conll2003/{file_out}"
    pp_conll2003(path_in, path_out)
