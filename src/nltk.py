import nltk
from nltk.corpus import brown
from nltk.tag import UnigramTagger

nltk.download("punkt", quiet=True)
nltk.download("brown", quiet=True)
nltk.download("tagsets", quiet=True)
nltk.download("universal_tagset", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


def most_common_func(pos_tags):
    tag_fd = nltk.FreqDist(tag for (word, tag) in pos_tags)
    return tag_fd.most_common()


def eval_tags(pos_tags):
    tagger = UnigramTagger(brown.tagged_sents(categories="news"))
    return tagger.evaluate([pos_tags])


def sentence_get_pos_tags(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags
