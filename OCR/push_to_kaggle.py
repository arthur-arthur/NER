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
    help="Embedding storage mode. When storage_mode is 'gpu', GPU will be automatically enabled in the Kaggle environment.",
)
@click.argument("emb", nargs=-1)  # multiple arguments for emb --> tuple
@click.argument("seeds", nargs=-1)  # TODO: nargs=-1 for emb argument eats all following arguments.
                                    # TODO: fix w/ config file
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

    build_script_from_options(data, lang, emb1, epochs, storage)
    pusher = KagglePusher(data, lang, emb1, storage)
    pusher.init_metadata()
    pusher.push()

push_to_kaggle()
