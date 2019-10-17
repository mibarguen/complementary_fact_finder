import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.corpora import WikiCorpus, MmCorpus, Dictionary
import pickle

class LoadWikiCorpus:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.wiki = wiki = WikiCorpus('data/simplewiki-latest-pages-articles.xml.bz2', tokenizer_func=tokenize)
        self.wiki.metadata = True
        self.sents = self.get_sents(1000)
        self.save_list(self.sents, 'data/sents.pkl')
        self.token_sents = self.tokenize_sents(self.sents)
        self.save_list(self.token_sents, 'data/token_sents.pkl')

    def get_first_paragraph(self, sents):
        s = []
        i = 0
        while (sents[i][0] != '=') and (i < len(sents) - 1):
            s.append(sents[i])
            i += 1
        return s

    def get_sents(self, num_articles):
        s = []
        i = 0
        for t in self.wiki.get_texts():
            title = t[1][1]
            content = t[0]
            sents = self.get_first_paragraph(content)
            clean_sents = [re.sub(r'[^\w\s]', '', sent) + '.' for sent in sents]
            s.extend([(title, sent) for sent in clean_sents])
            i += 1
            if i > num_articles:
                break
        return s

    def tokenize_sentence(self, sentence, stop_words, lemmatizer=None):
        sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        tokens = [w for w in word_tokenize(sentence) if not w in stop_words and len(w) > 2]
        if lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def tokenize_sents(self, sents):
        return [(self.tokenize_sentence(s[1], self.stop_words, self.lemmatizer)) for s in sents]

    def save_list(self, list_input, output_path):
        with open(output_path, 'wb') as fb:
            pickle.dump(list_input, fb)


