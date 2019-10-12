from gensim.corpora.wikicorpus import WikiCorpus


class LoadWiki:
    def __init__(self, link):
        self.link = link
        self.wiki = WikiCorpus(self.link)
        self.num_docs = 0
        self.output_file = 'data/wiki_corpus.txt'

    def load_text(self, t):
        self.num_docs += 1
        if self.num_docs % 100 == 0:
            print(f'{self.num_docs} processed. ')
        return bytes(' '.join(t), 'utf-8').decode('utf-8') + '\n'

    def write_file(self, t, output_write):
        text = self.load_text(t)
        output_write.write(text)
        return text

    def load_corpus(self):
        output_write = open(self.output_file, 'w')
        wiki_corp = [self.write_file(t, output_write) for t in self.wiki.get_texts()]
        output_write.close()
        return wiki_corp


if __name__ == '__main__':
    wiki = LoadWiki('data/simplewiki-latest-pages-articles.xml.bz2')
    wiki.load_corpus()