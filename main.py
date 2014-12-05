#!/usr/bin/env python

from Library import Corpus
from os import *


QUERY_INPUT_MESSAGE = "Enter a query, or hit enter to quit: "

if __name__ == '__main__':
    #corpus = Corpus("data/title_author_min.txt") # small file for testing
    corpus = Corpus("data/title_author.tab.txt")  # original big file
    print("[Main] Loading books...")
    corpus.load_books()
    
    print("[Main] Done loading books, %d docs in index" % corpus.books_count())
    corpus.build_term_index()
    print("[Main] Done Indexing terms.")
    corpus.buildTfidfRank()
    print("[Main] Done build Tfidf Rank.")
    query = None
    _NO_RESULTS_MESSAGE = "Sorry, no results."
    search_results= ""
    
    while query is not "":
       query = input(QUERY_INPUT_MESSAGE)
       if len(query) > 0:
           search_results = corpus.search_corpus(query, n_results=10)
           if len(search_results) > 0:
               print("\n Results are: \n")
               print("=======================================================================\n")
               for result in search_results: print("%s \n" % result)
               print("=======================================================================\n")
           else:
               print(_NO_RESULTS_MESSAGE)
           
           
    print("\n Thanks and have a good day!")
    