from gensim.corpora import Dictionary
from src.clean_text import TextCleaner
from gensim.models import TfidfModel, LsiModel, LdaModel
from gensim.similarities import MatrixSimilarity
from src.clean_tools import *
from abc import abstractmethod


class WikiModel:
    def __init__(self, wiki_tokens_path='data/token_sents.pkl', wiki_sents_path='data/sents.pkl',
                 student_tokens_path='data/children_data.json'):
        self.wiki_tokens = self.load_wiki(wiki_tokens_path)
        self.wiki_sents = self.load_wiki(wiki_sents_path)
        self.wiki_dict = Dictionary(self.wiki_tokens)
        self.wiki_corpus = [self.wiki_dict.doc2bow(t) for t in self.wiki_tokens]
        self.student_tokens = self.load_student(student_tokens_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.clean_text = TextCleaner()
        self.wiki_tfidf = TfidfModel(self.wiki_corpus, id2word=self.wiki_dict)
        self.wiki_tfidf_corpus = self.wiki_tfidf[self.wiki_corpus]

    def load_wiki(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_student(self, filepath):
        with open(filepath, 'rb') as f:
            return json.load(f)

    def get_keywords(self, essay):
        # remove punctuation.
        essay_clean = re.sub(r'[^\w\s]', '', essay)
        return get_keywords_from_sentence(essay_clean, self.lemmatizer, self.stop_words)

    def string_preds(self, preds):
        string_results = [(self.wiki_sents[pred[0]], pred[1]) for pred in preds]
        return string_results

    def process_sent(self, sent):
        clean_sent = re.sub(r'[^\w\s.:]', '', sent)
        return clean_sent

    def most_important_sent(self, article_sents):
        tfidf_scores = np.array([self.tfidf_score_sent(s) for s in article_sents])
        return self.process_sent(article_sents[np.argmax(tfidf_scores)])

    def predict_input(self, essay):
        essay_tokens = self.get_keywords(essay)
        self.predict(essay_tokens)

    def tfidf_score_sent(self, sent):
        return np.sum([t[1] for t in self.wiki_tfidf[
            self.wiki_dict.doc2bow(self.clean_text.get_keywords_from_sentence(sent))]])

    def most_important_sent(self, article_sents):
        tfidf_scores = np.array([self.tfidf_score_sent(s) for s in article_sents])
        return self.process_sent(article_sents[np.argmax(tfidf_scores)])

    @abstractmethod
    def predict(self, text, num_results=10, print_results=True):
        pass

    @abstractmethod
    def predict_index(self, text, num_results=10, print_results=True):
        pass


class LsiWikiModel(WikiModel):
    def __init__(self, num_topics=100, wiki_tokens_path='data/token_sents.pkl', wiki_sents_path='data/sents.pkl',
                 student_tokens_path='data/children_data.json'):
        super().__init__(wiki_tokens_path, wiki_sents_path, student_tokens_path)
        self.lsi = self.compute_lsi(num_topics)
        self.lsi_index = MatrixSimilarity(self.lsi[self.wiki_tfidf_corpus])

    def compute_lsi(self, num_topics=None):
        lsi = LsiModel(self.wiki_tfidf_corpus, num_topics=num_topics, id2word=self.wiki_dict)
        return lsi

    def predict(self, text, num_results=10, print_results=True):
        vec_bow = self.wiki_tfidf[self.wiki_dict.doc2bow(text)]
        vec_lsi = self.lsi[vec_bow]
        sims = self.lsi_index[vec_lsi]
        sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        preds = [(index, score) for index, score in sorted_sims[:num_results]]
        if print_results:
            for i in self.string_preds(preds):
                # print(f'Score: {round(i[1], 3)}')
                print('Topic: ', i[0][0])
                print(self.most_important_sent(i[0][1]))
                #print(i[0][1].replace('\n', ''))
                print()
        return preds

    def predict_index(self, student_index, num_results=10, print_results=True):
        text = self.student_tokens[student_index]
        return self.predict(text, num_results, print_results)


class LdaWikiModel(WikiModel):
    def __init__(self, num_topics=16, wiki_tokens_path='data/token_sents.pkl', wiki_sents_path='data/sents.pkl',
                 student_tokens_path='data/children_data.json'):
        super().__init__(wiki_tokens_path, wiki_sents_path, student_tokens_path)
        self.lda = self.compute_lda(num_topics)
        self.lda_index = MatrixSimilarity(self.lda[self.wiki_corpus])

    def compute_lda(self, num_topics=None):
        lda = LdaModel(self.wiki_corpus, num_topics=num_topics, id2word=self.wiki_dict)
        return lda

    def predict(self, text, num_results=10, print_results=True):
        vec_bow = self.wiki_dict.doc2bow(text)
        vec_lda = self.lda[vec_bow]
        sims = self.lda_index[vec_lda]
        sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        preds = [(index, score) for index, score in sorted_sims[:num_results]]
        if print_results:
            print(' '.join(text))
            print('\n\n')
            print('Results: \n')
            for i in self.string_preds(preds):
                print('Topic: ', i[0][0])
                print(self.most_important_sent(i[0][1]))
                print()
        return preds

    def predict_index(self, student_index, num_results=10, print_results=True):
        text = self.student_tokens[student_index]
        return self.predict(text, num_results, print_results)

