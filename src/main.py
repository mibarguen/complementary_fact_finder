from src.topic_models import *
import re


def main(model):
    print("Write 'exit' to escape program. Press enter and type '--facts' after you have written your essay to get results.")
    n = input("Copy-paste your essay here:    ")
    while True:
        current_input = input()
        n += current_input
        if current_input.lower() == 'exit':
            break
        elif '--facts' in current_input:
            print('Finding Facts... ')
            print('\nHere are some facts we think would make your essay more detailed.\n')
            n = re.sub(r'--facts', '', n)
            model.predict_input(n)
            print( "\n\nTo escape, type 'exit' or to find more facts from another essay, copy-paste your essay here and type --facts to get suggested facts:   ")
            n = ""


if __name__ == '__main__':
    print('Initializing model...')
    lsi_wiki = LsiWikiModel(student_tokens_path='data/student_keywords.json', wiki_tokens_path='data/articles_keywords.pkl', wiki_sents_path='data/articles.pkl')
    main(lsi_wiki)
