import re
from nltk.corpus import stopwords
from gensim.corpora import WikiCorpus
import pickle
from nltk.tokenize import sent_tokenize


class WikiCorpusLoader:
    def __init__(self, num_articles):
        self.stop_words = set(stopwords.words('english'))
        self.wiki = WikiCorpus('data/simplewiki-latest-pages-articles.xml.bz2', tokenizer_func=self.tokenize, processes=7)
        self.wiki.metadata = True
        self.sents = self.get_articles(num_articles)
        self.save_list(self.sents, 'data/articles.pkl')

    @staticmethod
    def tokenize(text, token_min_len, token_max_len, lower):
        return [token for token in sent_tokenize(text)
                if len(token) > 30 and not token.startswith('_')]

    def get_abstract(self, sents):
        s = []
        i = 0
        while (sents[i][0] != '=') and (i < len(sents) - 1):
            s.append(self.post_process_sents(sents[i]))
            i += 1
        return s

    def post_process_sents(self, sent):
        clean_text = re.sub(r'[^\w\s]', '', sent)
        clean_text = re.sub(r'\n', '', clean_text)
        return clean_text + '. '

    def get_articles(self, num_articles):
        s = []
        i = 0
        for t in self.wiki.get_texts():
            title = t[1][1]
            if 'List' in title:
                 print(title)
            else:
                content = t[0]
                clean_content = self.get_abstract(content)
                s.append((title, clean_content))
            i += 1
            if i % 20 == 0:
                print(i)
            if i > num_articles:
                print('Why here? ')
                break
        return s

    def save_list(self, list_input, output_path):
        with open(output_path, 'wb') as fb:
            pickle.dump(list_input, fb)


if __name__ == '__main__':
    wiki_loader = WikiCorpusLoader(5000)

