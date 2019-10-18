""" Functionality to get features for a set of 
"""

import spacy
import scipy
import numpy as np
import json

def cosine_similarity(arr1,arr2):
    """ Returns the cosine similarity between two vectors
    args:
        arr1, arr2 (arr): average word embeddings
    """
    return 1-scipy.spatial.distance.cosine(arr1,arr2)


def get_suggested_text():
    """ Returns suggested wiki sentences based on an ensemble of unsupervised methods.
    """ 
    # for now, just print the sentences
    pass


def get_top_wiki_sentences_ensemble():
    """ Getting the top sentences using an ensemble of lda, lsi, avg. word vec models.
    """
    pass


def get_word_vec_top_sentences(sentence_vec,wiki_fpath,num_output=10):
    """ Takes in a sentence and returns the most similar sentences within the wikipedia corpus.
    args:
        sentence_vec (np.array): shape(300,), average word embeddings for a sentence 
        wiki_fpath (str): path to the wikipedia json file holding embeddings
        num_output (int): number of best indices w/ scores to return
    """
    wiki_data = [] # save this as the data instead
    with open(wiki_fpath) as afile:
        data_lines = json.load(afile)    # loading in the wiki data
        for line in data_lines:
            wiki_data.append(np.array(line))

    wiki_array = np.array(wiki_data)
    euclidian = np.sqrt(np.sum(np.square(wiki_array-sentence_vec),axis=1)) # distance
    euclidian_list = list(euclidian)
    euclid_text = [(i,euclidian_list[i]) for i in range(len(euclidian_list))] # index of text & score
    euclid_text.sort(key=lambda tup: tup[1])
    return euclid_text[:num_output]

