from gensim.corpora.wikicorpus import WikiCorpus
from nltk.tokenize import sent_tokenize

class LoadWiki:
    def __init__(self, link):
        self.link = link
        self.wiki = WikiCorpus(self.link, tokenizer_func=self.tokenize, processes=7)
        print('here')
        # self.wiki = WikiCorpus.load('data/wiki_model.dict')
        self.wiki.save('data/wiki_model_sents.dict')
        self.num_docs = 0
        self.output_file = 'data/wiki_corpus.txt'

    def load_text(self, text):
        return ' '.join([t.decode("utf-8") for t in text])

    def write_file(self, t, output_write):
#        text = self.load_text(t)
        output_write.write('\n'.join(t))
        self.num_docs += 1
        if self.num_docs % 100 == 0:
            print(f'{self.num_docs} processed. ')
        return t

    def tokenize(self, text, token_min_len, token_max_len, lower):
        # override original method in wikicorpus.py
        return [token for token in sent_tokenize(text)
                if len(token) >= 20 and not token.startswith('_')]

    def load_corpus(self):
        output_write = open(self.output_file, 'w')
        wiki_corp = [self.write_file(t, output_write) for t in self.wiki.get_texts()]
        output_write.close()
        return wiki_corp


if __name__ == '__main__':
    wiki = LoadWiki('data/simplewiki-latest-pages-articles.xml.bz2')
    wiki.load_corpus()