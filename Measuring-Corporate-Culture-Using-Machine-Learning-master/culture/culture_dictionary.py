import itertools
import math
import os
import pickle
import statistics as s
from collections import Counter, OrderedDict, defaultdict
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter

import gensim
import numpy as np
import pandas as pd
import tqdm
from sklearn import preprocessing


def expand_words_dimension_mean(
    word2vec_model,
    seed_words,
    n=50,
    restrict=None,
    min_similarity=0,
    filter_word_set=None,
):
    """For each dimensional mean vector, search for the closest n words"""
    vocab_number = len(word2vec_model.wv.key_to_index)  # Adjusted to use key_to_index
    expanded_words = {}
    all_seeds = set(itertools.chain.from_iterable(seed_words.values()))  # More efficient gathering of all seed words

    if restrict is not None:
        restrict = int(vocab_number * restrict)

    for dimension, seeds in seed_words.items():
        valid_seeds = [word for word in seeds if word in word2vec_model.wv.key_to_index]  # Check for word presence correctly

        if valid_seeds:
            similar_words = word2vec_model.wv.most_similar(
                positive=valid_seeds, topn=n, restrict_vocab=restrict)
            similar_words = [word for word, similarity in similar_words
                             if similarity >= min_similarity and word not in all_seeds and (filter_word_set is None or word not in filter_word_set)]
        else:
            similar_words = []

        expanded_words[dimension] = set(similar_words)

    return expanded_words


def rank_by_sim(expanded_words, seed_words, model):
    """ Rank each dim in a dictionary based on similarity to the seed words mean
    Returns: expanded_words_sorted {dict[str:list]}
    """
    expanded_words_sorted = dict()
    for dimension, words in expanded_words.items():
        # Filter valid seed words present in the model's vocabulary
        dimension_seed_words = [
            word for word in seed_words[dimension] if word in model.wv.key_to_index
        ]
        # Compute similarity and sort words
        if dimension_seed_words:  # Ensure there are valid seed words
            similarity_dict = {
                word: model.wv.n_similarity(dimension_seed_words, [word])
                for word in words if word in model.wv.key_to_index
            }
            sorted_similarity_list = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
            expanded_words_sorted[dimension] = [word for word, sim in sorted_similarity_list]
        else:
            expanded_words_sorted[dimension] = []

    return expanded_words_sorted



def write_dict_to_csv(culture_dict, file_name):
    """write the expanded dictionary to a csv file, each dimension is a column, the header includes dimension names
    
    Arguments:
        culture_dict {dict[str, list[str]]} -- an expanded dictionary {dimension: [words]}
        file_name {str} -- where to save the csv file?
    """
    pd.DataFrame.from_dict(culture_dict, orient="index").transpose().to_csv(
        file_name, index=None
    )


def read_dict_from_csv(file_name):
    """Read culture dict from a csv file

    Arguments:
        file_name {str} -- expanded dictionary file
    
    Returns:
        culture_dict {dict{str: set(str)}} -- a culture dict, dim name as key, set of expanded words as value
        all_dict_words {set(str)} -- a set of all words in the dict
    """
    print("Importing dict: {}".format(file_name))
    culture_dict_df = pd.read_csv(file_name, index_col=None)
    culture_dict = culture_dict_df.to_dict("list")
    for k in culture_dict.keys():
        culture_dict[k] = set([x for x in culture_dict[k] if x == x])  # remove nan

    all_dict_words = set()
    for key in culture_dict:
        all_dict_words |= culture_dict[key]

    for dim in culture_dict.keys():
        print("Number of words in {} dimension: {}".format(dim, len(culture_dict[dim])))

    return culture_dict, all_dict_words


def deduplicate_keywords(word2vec_model, expanded_words, seed_words):
    """Ensure a word is attributed to only one dimension based on highest similarity to seed words."""
    word_counts = Counter(word for words in expanded_words.values() for word in words)

    # Filter out unique words directly
    unique_words = {word for word, count in word_counts.items() if count == 1}
    for words in expanded_words.values():
        words.intersection_update(unique_words)

    # Process duplicated words
    duplicated_words = {word for word, count in word_counts.items() if count > 1}
    for word in duplicated_words:
        best_dimension = max(expanded_words, key=lambda dim: (word in expanded_words[dim],
                                                              word2vec_model.wv.n_similarity(seed_words[dim], [word])))
        for dim in expanded_words:
            if word in expanded_words[dim] and dim != best_dimension:
                expanded_words[dim].remove(word)

    return expanded_words


def score_one_document_tf(document, expanded_words, list_of_list=False):
    """score a single document using term freq, the dimensions are sorted alphabetically
    
    Arguments:
        document {str} -- a document
        expanded_words {dict[str, set(str)]} -- an expanded dictionary
    
    Keyword Arguments:
        list_of_list {bool} -- whether the document is splitted (default: {False})
    
    Returns:
        [int] -- a list of : dim1, dim2, ... , document_length
    """
    if list_of_list is False:
        document = document.split()
    dimension_count = OrderedDict()
    for dimension in expanded_words:
        dimension_count[dimension] = 0
    c = Counter(document)
    for pair in c.items():
        for dimension, words in expanded_words.items():
            if pair[0] in words:
                dimension_count[dimension] += pair[1]
    # use ordereddict to maintain order of count for each dimension
    dimension_count = OrderedDict(sorted(dimension_count.items(), key=lambda t: t[0]))
    result = list(dimension_count.values())
    result.append(len(document))
    return result


def score_tf(documents, document_ids, expanded_words, n_core=1):
    """score using term freq for documents, the dimensions are sorted alphabetically
    
    Arguments:
        documents {[str]} -- list of documents
        document_ids {[str]} -- list of document IDs
        expanded_words {dict[str, set(str)]} -- dictionary for scoring
    
    Keyword Arguments:
        n_core {int} -- number of CPU cores (default: {1})
    
    Returns:
        pandas.DataFrame -- a dataframe with columns: Doc_ID, dim1, dim2, ..., document_length
    """
    if n_core > 1:
        pool = Pool(n_core)  # number of processes
        count_one_document_partial = partial(
            score_one_document_tf, expanded_words=expanded_words, list_of_list=False
        )
        results = list(pool.map(count_one_document_partial, documents))
        pool.close()
    else:
        results = []
        for i, doc in enumerate(documents):
            results.append(
                score_one_document_tf(doc, expanded_words, list_of_list=False)
            )
    df = pd.DataFrame(
        results, columns=sorted(list(expanded_words.keys())) + ["document_length"]
    )
    df["Doc_ID"] = document_ids
    return df


def score_tf_idf(
    documents,
    document_ids,
    expanded_words,
    df_dict,
    N_doc,
    method="TFIDF",
    word_weights=None,
    normalize=False,
):
    """Calculate tf-idf score for documents

    Arguments:
        documents {[str]} -- list of documents (strings)
        document_ids {[str]} -- list of document ids
        expanded_words {{dim: set(str)}}} -- dictionary
        df_dict {{str: int}} -- a dict of {word:freq} that provides document frequencey of words
        N_doc {int} -- number of documents

    Keyword Arguments:
        method {str} -- 
            TFIDF: conventional tf-idf 
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict 
            (default: {TFIDF})
        normalize {bool} -- normalized the L2 norm to one for each document (default: {False})
        word_weights {{word:weight}} -- a dictionary of word weights (e.g. similarity weights) (default: None)

    Returns:
        [df] -- a dataframe with columns: Doc_ID, dim1, dim2, ..., document_length
        [contribution] -- a dict of total contribution (sum of scores in the corpus) for each word 
    """
    print("Scoring using {}".format(method))
    contribution = defaultdict(int)
    results = []
    for i, doc in enumerate(tqdm.tqdm(documents)):
        document = doc.split()
        dimension_count = OrderedDict()
        for dimension in expanded_words:
            dimension_count[dimension] = 0
        c = Counter(document)
        for pair in c.items():
            for dimension, words in expanded_words.items():
                if pair[0] in words:
                    if method == "WFIDF":
                        w_ij = (1 + math.log(pair[1])) * math.log(
                            N_doc / df_dict[pair[0]]
                        )
                    elif method == "TFIDF":
                        w_ij = pair[1] * math.log(N_doc / df_dict[pair[0]])
                    elif method == "TFIDF+SIMWEIGHT":
                        w_ij = (
                            pair[1]
                            * word_weights[pair[0]]
                            * math.log(N_doc / df_dict[pair[0]])
                        )
                    elif method == "WFIDF+SIMWEIGHT":
                        w_ij = (
                            (1 + math.log(pair[1]))
                            * word_weights[pair[0]]
                            * math.log(N_doc / df_dict[pair[0]])
                        )
                    else:
                        raise Exception(
                            "The method can only be TFIDF, WFIDF, TFIDF+SIMWEIGHT, or WFIDF+SIMWEIGHT"
                        )
                    dimension_count[dimension] += w_ij
                    contribution[pair[0]] += w_ij / len(document)
        dimension_count = OrderedDict(
            sorted(dimension_count.items(), key=lambda t: t[0])
        )
        result = list(dimension_count.values())
        result.append(len(document))
        results.append(result)
    results = np.array(results)
    # normalize the length of tf-idf vector
    if normalize:
        results[:, : len(expanded_words.keys())] = preprocessing.normalize(
            results[:, : len(expanded_words.keys())]
        )
    df = pd.DataFrame(
        results, columns=sorted(list(expanded_words.keys())) + ["document_length"]
    )
    df["Doc_ID"] = document_ids
    return df, contribution


def compute_word_sim_weights(file_name):
    """Compute word weights in each dimension.
    Default weight is 1/ln(1+rank). For example, 1st word in each dim has weight 1.44,
    10th word has weight 0.41, 100th word has weigh 0.21.
    
    Arguments:
        file_name {str} -- expanded dictionary file
    
    Returns:
        sim_weights {{word:weight}} -- a dictionary of word weights
    """
    culture_dict_df = pd.read_csv(file_name, index_col=None)
    culture_dict = culture_dict_df.to_dict("list")
    sim_weights = {}
    for k in culture_dict.keys():
        culture_dict[k] = [x for x in culture_dict[k] if x == x]  # remove nan
    for key in culture_dict:
        for i, w in enumerate(culture_dict[key]):
            sim_weights[w] = 1 / math.log(1 + 1 + i)
    return sim_weights
