# Combine datasets (full and random sample of 1//n_sentences per partition
import random


def combine(directories, path_out, reduce):

    combined = []

    for partition in ["train.txt", "dev.txt", "test.txt"]:

        for d in directories:

            path = f"{d}/{partition}"

            out = []
            sentence = ""

            with open(path, "r") as read:

                for line in read:

                    if line != "\n":
                        sentence += line
                    else:
                        if sentence:
                            out.append(sentence)
                            sentence = ""

            if reduce:
                out = random.sample(out, len(out) // len(directories))

            combined.extend(out)

        with open(f"{path_out}/{partition}", "w") as f:
            print("\n".join(combined), file=f)


directories = ["conll2003", "conll2002", "wikiner"]

for reduce, path_out in [
    (False, "trilingual/full"),
    (True, "trilingual/reduced"),
]:

    combine(directories=directories, path_out=path_out, reduce=reduce)
