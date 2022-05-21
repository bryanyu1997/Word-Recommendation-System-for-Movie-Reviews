import scipy
import numpy as np

from datetime import datetime
from numpy import dot
from numpy.linalg import norm
from itertools import product

from src.nltk import sentence_get_pos_tags
from src.hugging_face import hugging_face_embedding


def find_all_words(sentence_tags, query_tag):
    query_words_array = np.array(sentence_tags[0])
    query_words = query_words_array[query_words_array[:, 1] == query_tag, 0]

    # combine
    words_array = np.array(sentence_tags[1:])
    words = words_array[words_array[:, 1] == query_tag, 0]

    return np.array(list(set(words))), np.array(list(set(query_words)))


def find_all_words_multi(sentence_tags, query_tag):
    words = list()
    query_words = list()
    for i, sentence_tag in enumerate(sentence_tags):
        for word, tag in sentence_tag:
            if query_tag == tag and True:  # query sentence
                if i == 0:
                    query_words.append(word)
                else:
                    words.append(word)
    return words, query_words


def search_candidate_words(query, sentence_list, pos_dataset=None):
    query_sentence = query[0]
    query_tags = query[1]

    # div sentences
    query_words_dict = dict()
    words_dict = dict()
    for query_tag in query_tags:
        # {tag: np.array([("words", 0], ...]), ...}
        query_sentence_tags = sentence_get_pos_tags(query_sentence)
        query_words_dict[query_tag] = list(set([(word[0], i) \
            for i, word in enumerate(query_sentence_tags) \
            if word[1] == query_tag]))
        sentence_tags = []
        for sentence in sentence_list:
            if pos_dataset and sentence in pos_dataset:
                sentence_tags.extend([
                    word[0] for word in pos_dataset[sentence]
                    if word[1] == query_tag])
            else:
                sentence_tags.extend([
                    word[0] for word in sentence_get_pos_tags(sentence)
                    if word[1] == query_tag])
        words_dict[query_tag] = list(set(sentence_tags))

    return words_dict, query_words_dict, query_sentence_tags


def search_candidate_words_slow(query, sentence_list, pos_dataset=None):
    query_sentence = query[0]
    query_tag = query[1]

    # div sentences
    sentence_tags = [sentence_get_pos_tags(query_sentence)]
    for sentence in sentence_list:
        if pos_dataset and sentence in pos_dataset:
            sentence_tags.extend([
                word for word in pos_dataset[sentence]
                if word[1] == query_tag])
        else:
            sentence_tags.extend([
                word for word in sentence_get_pos_tags(sentence)
                if word[1] == query_tag])

    # find all words
    words, query_words = find_all_words(sentence_tags, query_tag)

    return words, query_words


def word2vec_array(w2v_model, words_dict, query_mode=False):
    embedding_matrix_dict = dict()
    for tag, words in words_dict.items():
        embedding_matrix = list()
        for i, word in enumerate(words):
            if query_mode:
                idx = word[1]
                word = word[0]
            if word in w2v_model:
                embedding_vector = w2v_model[word]
                if embedding_vector is not None:
                    if query_mode:
                        embedding_matrix.append([embedding_vector, word, idx])
                    else:
                        embedding_matrix.append([embedding_vector, word])

        embedding_matrix_dict[tag] = embedding_matrix

    return embedding_matrix_dict


def eval_distance(query,query_sentence_tags, array_query_dict,
                  array_words_dict, method="cosine", only_word=False):
    candidate_words_distances_dict = dict()
    for tag, array_query_idx in array_query_dict.items():
        array_words = array_words_dict[tag]
        words_distances = dict()
        for a_query in array_query_idx:
            key_word = find_keyword(query, a_query, query_sentence_tags,
                                    only_word=only_word)
            words_distances[key_word] = list()
            for array_word in array_words:
                if method == "cosine":
                    words_distances[key_word].append([dot(
                        a_query[0], array_word[0]) / (norm(a_query[0]) * 
                            norm(array_word[0])), array_word[1]])
                elif method == "kl":
                    words_distances[key_word].append([
                        sum(scipy.special.kl_div(a_query[0], array_word[0])),
                        array_word[1]])
                elif method == "l2":
                    words_distances[key_word].append([
                        1 - (pow(sum((a_query[0] - array_word[0]) ** 2), 0.5)
                            / a_query[0].shape[0]), array_word[1]])
            words_distances[key_word].sort(key=lambda k: k[0], reverse=True)

            # delete same word
            if words_distances[key_word] and \
                words_distances[key_word][0][1] == a_query[1]:
                del words_distances[key_word][0]

        candidate_words_distances_dict[tag] = words_distances

    return candidate_words_distances_dict


def find_keyword(query, a_query, query_sentence_tags, only_word=False):
    if only_word:
        return a_query[1]
    key_word = ''
    search_sentence = query[0]
    for i, word_tag in enumerate(query_sentence_tags[:a_query[2] + 1]):
        if word_tag[0] in search_sentence:
            inx = search_sentence.index(word_tag[0])
            if i == len(query_sentence_tags[:a_query[2] + 1]) - 1:
                key_word = key_word + search_sentence[:inx] + '[MASK]' + \
                            search_sentence[inx + len(word_tag[0]):]
            else:
                key_word = key_word + search_sentence[: inx + len(word_tag[0])]
                search_sentence = search_sentence[inx + len(word_tag[0]):]

    return key_word


def final_wordsandquery_dict(query_words_dict, cand_words_distances_dict,
                             n_largest):
    final_query_words_dict = {tag: list()
            for tag, words_idx in query_words_dict.items()}
    final_words = {tag: list() for tag, words_idx in query_words_dict.items()}
    for tag, cand_words_distance in cand_words_distances_dict.items():
        for query_word, word_dists in cand_words_distance.items():
            idx = np.where(
                np.array(query_words_dict[tag])[:, 0] == query_word)[0]
            if idx.shape[0] != 0:
                final_query_words_dict[tag].append(
                    query_words_dict[tag][idx[0]])
                final_words[tag].append([dw[1]
                    for dw in word_dists[:min(n_largest, len(word_dists))]])

    return final_query_words_dict, final_words


def hugging_face_distance(model, tokenizer, query, query_sent_tags,
        query_words, words_dict, n_largest, GPU, method="cosine"):

    # get final sentences to compute
    total_cands = [[word[0]] for word in query_sent_tags]
    for tag, word_idx in query_words.items():
        total_cands = [tword if i not in [t[1] for t in word_idx]
                else words_dict[tag][[t[1] for t in word_idx].index(i)]
                for i, tword in enumerate(total_cands)]
    sents = [query[0]] + [" ".join(list(sent)) for sent in product(*total_cands)]

    # get embeddings of sentences
    sents_emb = hugging_face_embedding(model, tokenizer, sents, GPU)

    # compute distance between sentences
    if method == "cosine":
        words_dists = [dot(sents_emb[0], embedding) / (
            norm(sents_emb[0]) * norm(embedding))
            for embedding in sents_emb[1:]]
    elif method == "kl":
        words_dists = [sum(scipy.special.kl_div(sents_emb[0], embedding))
            for embedding in sents_emb[1:]]
    elif method == "l2":
        words_dists = [1 - (math.sqrt(
            sum((sents_emb[0] - embedding) ** 2)) / sents_emb[0].shape[0])
            for embedding in sents_emb[1:]]

    # get ranks of candidate sentences
    rank_idx = sorted(range(len(words_dists)),
                      key=lambda k: words_dists[k], reverse=True)
    final_cands = [
        [words_dists[idx], sents[idx + 1]]
        for idx in rank_idx[:min(len(rank_idx), n_largest)]]

    return {query[0]: final_cands}


def compute(prediction, ground_truth, num=100):
    pred_set = set(prediction)
    gt_set = set(ground_truth)
    common = pred_set.intersection(gt_set)
    preds = [pred for pred in prediction if pred in common]
    gts = [gt for gt in ground_truth if gt in common]
    total = len(preds)
    diffs = 0
    for i, gt in enumerate(gts):
        order_in_gts = i
        order_in_preds = preds.index(gt)
        order_in_pred_set = list(pred_set).index(gt)
        diff = abs(order_in_gts - order_in_preds)
        diff_absol = 0 if abs(order_in_gts - order_in_pred_set) < 5 else \
                abs(order_in_gts - order_in_pred_set)
        diffs += diff + diff_absol / (num - 5)
    diffs = diffs + 0.1 * (1 - total / 5)
    return diffs, total


def print_log(message):
    print("[{}] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           message))
