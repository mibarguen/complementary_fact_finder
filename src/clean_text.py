import json
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
import spacy
import numpy as np
from nltk.util import ngrams

class TextCleaner:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_keywords_from_sentence(self, sent):
        """ Gets the keywords from a given sentence, defined as Nouns and Verbs, and lemamtizing them
        """
        tokens = word_tokenize(sent)
        pos_tokens = pos_tag(tokens)
        keep_tokens = [self.clean_token(token_tup[0].lower()) for token_tup in pos_tokens if
                       (("NN" in token_tup[1] or "VB" in token_tup[1]) and (token_tup[0]).lower() not in self.stop_words) and (len(token_tup[0]) > 2)]
        keep_tokens = [self.lemmatizer.lemmatize(token) for token in keep_tokens]
        return keep_tokens

    def clean_token(self, token):
        token_clean = re.sub(r'[^\w\s]', '', token)
        token_clean = re.sub(r'[0-9]', '', token_clean)
        return token_clean


if __name__ == '__main__':
    text_cleaner = TextCleaner()
    with open('data/articles.pkl', 'rb') as f:
        wiki_articles = pickle.load(f)

    print(wiki_articles)
    cleaned_articles = [text_cleaner.get_keywords_from_sentence(''.join(s[1])) for s in wiki_articles]

    with open('data/articles_keywords.pkl', 'wb') as f:
        pickle.dump(cleaned_articles, f)

    with open('data/student_keywords.json', 'rb') as f:
        children_sents = json.load(f)
    keywords = children_sents


    def add_bigrams(keywords):
        sents = []
        for sent in keywords:
            sents.append(sent + list([str(n[0] + '_' + n[1]) for n in ngrams(sent, 2)]))
            # sents.append(list([str(n[0] + '_' + n[1]) for n in ngrams(sent, 2)]))

        return sents


    with open('data/student_keywords_bi.json', 'w') as f:
        bigrams = add_bigrams(children_sents)
        print(bigrams)
        json.dump(bigrams, f)

    with open('data/wiki_keywords_bi.pkl', 'wb') as f:
        bigrams = add_bigrams(cleaned_articles)
        pickle.dump(bigrams, f)