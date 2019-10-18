from src.topic_models import *


def main():
    print("Write 'exit' to escape program. Type --facts after you have written your essay to get results.")
    n = input("Copy-paste your essay here:    ")
    while True:
        current_input = input()
        n += current_input
        if current_input.lower() == 'exit':
            break
        elif '--facts' in current_input:
            print('Finding Facts... ')
            print('\n\nResults: ')
            print(n.replace('--find-facts', ''))
            print( "\n\nTo escape, type 'exit' or to find more facts from another essay, copy-paste your essay here and type --facts to get suggested facts:   ")
            n = ""


if __name__ == '__main__':
    print('Initializing model...')
    lsi_wiki = LsiWikiModel(student_tokens_path='data/student_keywords.json', wiki_tokens_path='data/wiki_keywords.pkl')
    main()