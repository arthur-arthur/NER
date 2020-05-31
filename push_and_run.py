import click
from utils import KagglePusher, build_script_from_options


@click.command()
@click.option(
    "--data", type=str, help="Dataset: [conll2002, conll2003, wikiner, trilingual]"
)
@click.option(
    "--lang", type=str, help="Language of embedding class: [en, nl, fr, multi]"
)
@click.option(
    "--epochs", type=int, default=100, show_default=True, help="Max number of epochs"
)
@click.option(
    "--storage",
    type=str,
    default="cpu",
    show_default=True,
    help="Embedding storage mode. When set to gpu, the script will also be run on a gpu",
)
@click.argument("emb", nargs=-1)  # multiple arguments for emb --> tuple
def push_to_kaggle(data, lang, emb, epochs, storage):

    """
    Embedding codes can be passed as arguments, the appropriate classes are initiated
    according to the --lang option. Multiple embeddings are concatenated (StackedEmbeddings)

    Abbreviations (not case-sensitive):

    =====================================================================

        "bert":     BERTje (nl), CamemBERT (fr), BERT (en), mBERT (multi)

        "bpe":      BytePairEmbeddings (fr, nl, en or multi)

        "ohe":      OneHotEmbeddings

        "char":     CharacterEmbeddings

        "ft":       fastText nl/fr/en (WordEmbeddings)

        "flair":    flair nl/fr/en, both fw and bw

    =====================================================================

    """

    build_script_from_options(data, lang, emb, epochs, storage)
    pusher = KagglePusher(data, lang, emb, storage)
    pusher.init_metadata()
    pusher.push()


push_to_kaggle()
