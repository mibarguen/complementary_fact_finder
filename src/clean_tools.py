""" Helper functions for cleaning the data. 
    Also handles saving post-engineered features to disk.
"""

import json
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
import spacy
import numpy as np
from spellchecker import SpellChecker

# "../../csc482_project_1/tai-documents-v3/tai-documents-v3.json"
def load_student_text(file_path, remove_bad_punc=False):
    """ Loads in the student text and returns a list of documents for each essay 
    args:
        file_path (str): path to the student json
        remove_bad_punc (bool): indicator if you want to remove mispelled words
    """
    text_file = open(file_path)
    texts_list = json.load(text_file)
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    spell = SpellChecker()
    all_student_words = [] # stores list of tokens for each sentence
    for text_dict in texts_list:
        raw_text = text_dict['plaintext']
        sents = sent_tokenize(raw_text)
        all_words = []
        for sent in sents:
            sent_tokens = tokenize_sentence(sent,stop_words,wordnet_lemmatizer)
            if remove_bad_punc:
                sent_tokens = [token for token in sent_tokens if len(spell.unknown([token]))==0]
            all_words += sent_tokens

        all_student_words.append(all_words)
    return all_student_words


def tokenize_sentence(sentence, stop_words, lemmatizer=None):
    """ Returns the tokens for a sentence as a list.
    """
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    tokens = [w for w in word_tokenize(sentence) if not w in stop_words and len(w) > 2]
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def average_word_vectors_feature_extractor(tokenized_file,out_file):
    """ Running through tokenized wiki sentences and calling feature engineering process. Saves engineered features to disk.
    args:
        tokenized_file: path to the json file of tokenized sentences
        out_file: path for saving the engineered features
    """ 
    nlp = spacy.load("en_core_web_lg")
    with open(tokenized_file, 'rb') as data_file:
        wiki_data = pickle.load(data_file)
    all_features = []
    for word_list in wiki_data:
        feature = get_average_word_vectors_features_list(word_list,nlp)
        all_features.append(list(feature))

    assert(len(all_features)==len(wiki_data))
    with open(out_file, 'w') as outfile: # saving the features
        json.dump(all_features, outfile)
    

def get_average_word_vectors_features_list(tokenized_list,nlp):
    """ Creates features for each of the wiki sentences using average word embeddings. Word2Vec.
    args:
        tokenized_list: tokenized list of words
        nlp: language model
    """
    n=0
    feature = np.zeros((300,))
    for word in tokenized_list:
        token = nlp(word)
        if token.has_vector:
            n += 1
            feature += token.vector
            assert(feature.shape==(300,))
    if n != 0:
        feature /= n
    return feature











