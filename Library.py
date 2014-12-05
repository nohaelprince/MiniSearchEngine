# -*- coding: utf-8 -*-
import os
import re
import ctypes
import math
import unicodedata
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
from scipy.sparse.sparsetools import csr_scale_rows, csr_scale_columns, \
        bsr_scale_rows, bsr_scale_columns 
from collections import defaultdict
from book import Book 

class Corpus():
#=================================================================   
     def __init__(self, fname):
         self.fname = fname    # file name containing book data.
         self.Books= []        # empty list of books (objects)
         self.term_index = defaultdict(list)  # dictionary <word, (list of all 
                                                  # the books id that contain the word)>
         self.stop_words = {}
         #self.rank = TfidfRank()
         
         self. vocabulary = defaultdict(int) # data dictionary <unique word: freq. of occurence per corpus>   
         self.tf_idf_matrix= []  # Matrix containing the ft-idf score for each term
                                 #in the corpus respecting the index stored in `vocabulary`.
         self.set_stop_words()    # assign value to self.stop_words dict. <stopword, True>'
         self.ft_matrix = []  # Matrix containing the frequency term for each term
                        # in the corpus respecting the index stored in `vocabulary`
#=================================================================            
     def books_count(self):
         return len(self.Books)
#=================================================================    
     @staticmethod
     def clean(line):
         
         _EXTRA_SPACE_REGEX = re.compile(r'\s+', re.IGNORECASE)
         _SPECIAL_CHAR_REGEX = re.compile(
                 # detect punctuation characters
                 r'(?P<p>(\.+)|(\?+)|(!+)|(:+)|(;+)|'
                 # detect special characters
                 r'(\(+)|(\)+)|(\}+)|(\{+)|("+)|(-+)|(\[+)|(\]+)|'
                 # detect commas NOT between numbers
                 r'(?<!\d)(,+)(?!=\d)|(\$+))')
                 
         line = line.lower()
         # All non-accents are removed - unicode used in python 2.7
         #line = unicodedata.normalize('NFD', unicode(line,"utf-8")).\
         #encode('ascii', 'ignore')
         line = unicodedata.normalize('NFD', str(line))
         # Special characters are replaced by whitespaces (i.e. -, [, etc.)
         line = line.replace("\t", "|").strip()
         # Punctuation marks are removed
         line = _SPECIAL_CHAR_REGEX.sub(' ', line)
         # Additional whitespaces between replaced by only one whitespaces
         line = _EXTRA_SPACE_REGEX.sub(' ', line)
         
         return line.split('|')
#=================================================================             
     def load_books(self):  # load books from fname
          print("[Corpus] Loading books from file...")
          #count = 0
          
          with open(self.fname) as infile:
              for line in infile:
                  cleanedLine = Corpus.clean(line) #call static function clean
                  
                  #print(cleanedLine)
                  # Tokenize
                  bookID = cleanedLine[0].strip()
                  title  = cleanedLine[1].strip()
                  author = cleanedLine[2].strip()
                  metadata = ' '.join(cleanedLine[1:])
                  
                  # insert book in the list of books of the Library
                  self.Books.append(Book(bookID, title, author, metadata))
                  
                  #count += 1
                  #if count>4: break # just read the first 5 entries as a test
                  #print book
# ===============================================================
     def set_stop_words(self): # assign value to self.stop_words
        current_path = os.path.dirname(os.path.realpath(__file__))
        STOP_WORDS_FILENAME = os.path.join(current_path, "data/stop_words.txt")
        
        # Stop words are words which are filtered out prior to processing of natural language data (e.g. the, is). The list of stop words is in `data/stop_words.txt` file.
        with open(STOP_WORDS_FILENAME) as stop_words_file:
            for word in stop_words_file:
                 self.stop_words[word.strip()] = True
#=================================================================                 
     def build_term_index(self): 
        """ build term_index (dict) where term is the key and a list
            of the IDs of all the books (documents) containing the key/term as values"""
        
        print("[Corpus] Start Indexing terms...")
        #for pos, book in enumerate(self.Books):
        #    print(book.get_words_count())
        
        for pos, book in enumerate(self.Books):
            l = book.get_words_count() 
            for word, count in l.items():
                if word not in self.stop_words:
                    self.term_index[word].append(pos)
            
        
        #print("self.term_index = %r" %self.term_index)
#=================================================================
     def build_vocabulary(self):
        """ Dictionary containing unique words in the corpus as
          keys and their respective global index as values 
          (used in tf-idf data structures) """
        vocabulary_index = 0
        for pos, book in enumerate(self.Books):
           l = book.get_words_count() 
           for word, count in l.items():
               if word not in self.vocabulary: #remove duplicate words over the corpus
                   self.vocabulary[word]= vocabulary_index
                   vocabulary_index += 1 
#=================================================================           
     def sparse_max_row(self): 
         """ if your matrix, lets call it a, is stored in CSR format, 
            then a.data has all the non-zero entries ordered by rows, 
            and a.indptr has the index of the first element of every row.
            
            returns: the max of each row for a large sparse matrix
        """
         ret = np.maximum.reduceat(self.ft_matrix.data, self.ft_matrix.indptr[:-1])
         ret[np.diff(self.ft_matrix.indptr) == 0] = 0
         return ret
#=================================================================
     def buildTfidfRank(self): # build tf-idf score for terms in the corpus
        """  Tf-idf stands for term-frequency times inverse document-frequency
          and it is a common term weighting scheme in document classification.
          The goal of using tf-idf instead of the raw frequencies of occurrence of a
          token in a given document is to scale down the impact of tokens that occur
          very frequently in a given corpus and that are hence empirically less
          informative than features that occur in a small fraction of the training
          corpus.
        """
        smoothing = 1   # Smoothing parameter for tf-idf computation
            # preventing by-zero divisions when a term does not occur in corpus 
        
       
        ifd_diag_matrix = [] # Vector containing the inverse document frequency for each
                         # term in the corpus. It respects the index stored in `vocabulary`
        
        print("[Corpus] Start ranking documents...")
        
        ## Build ft matrix
        self.build_vocabulary()
        #print(self.vocabulary)
        n_terms   = len(self.vocabulary)
        n_docs    = len(self.Books)
        
        #  Construct an empty sparse matrix (Row-based linked list sparse matrix) with shape (n_docs, n_terms) dtype is optional, defaulting to dtype=’d’.
        #  lil_matrix: linked list format 
        self.ft_matrix = sp.lil_matrix((n_docs, n_terms), dtype= np.int64) #dtype: data type of the matrix
        
        
        print("[Corpus] Vocabulary assembled with terms count %s" % n_terms)
        print("[Corpus] Starting term frequency computation...") 
        
        
        for pos, book in enumerate(self.Books):
            l = book.get_words_count()
            for word, count in l.items():
               word_index = self.vocabulary[word]    
               self.ft_matrix[pos, word_index] = book.count_for_word(word)
        
        #print(ft_matrix)
        
        self.ft_matrix = self.ft_matrix.tocsc() # Return a copy of ft_matrix in CSC (Compressed Sparse Column) format
                                           # Duplicate entries will be summed together.
        
        
        #print(self.ft_matrix)
        print("[Corpus] Starting tf-idf computation...") 
        # compute idf with smoothing
        df = np.diff(self.ft_matrix.indptr) +  smoothing
        n_docs_smooth = n_docs + smoothing
        
        # create diagonal matrix to be multiplied with ft
        idf = np.log(float(n_docs_smooth) / df) + 1.0
        self.ifd_diag_matrix = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)
        
        # compute tf-idf
        self.tf_idf_matrix = self.ft_matrix * self.ifd_diag_matrix
        self.tf_idf_matrix = self.tf_idf_matrix.tocsr()
        
        # compute td-idf normalization
        norm = self.tf_idf_matrix.tocsr(copy=True)
        norm.data **= 2
        norm = norm.sum(axis=1)
        n_nzeros = np.where(norm > 0)
        norm[n_nzeros] = 1.0 / np.sqrt(norm[n_nzeros])
        norm = np.array(norm).T[0]
        sp.sparsetools.csr_scale_rows(self.tf_idf_matrix.shape[0],
                                      self.tf_idf_matrix.shape[1],
                                      self.tf_idf_matrix.indptr,
                                      self.tf_idf_matrix.indices,
                                      self.tf_idf_matrix.data, norm)
        
        #print(self.tf_idf_matrix)
               
        """
        ft_matrix_maxTermFreq_perDoc = self.sparse_max_row() #get the maximum value for each row in CSR matrix (ft_matrix)
        #print(ft_matrix_maxTermFreq_perDoc)
        
        
        tf = 0.5 + (0.5 * (self.ft_matrix/ft_matrix_maxTermFreq_perDoc))
        #print(tf)
        
        ## compute idf (inverse document frequency) with smoothing
        # find total no. of documents containing the term; called (term_total_docs)
        term_total_docs = []
        for term, value in enumerate(self.term_index):
             term_total_docs.append(math.fabs(len(value) + smoothing))
             
        
        idf1 = math.log(n_docs, 10)
        for y in term_total_docs: idf2 = math.log(y,10)
        idf = idf1 - idf2 
        
        # compute tf-idf
        self.tf_idf_matrix = tf * idf
        print(self.tf_idf_matrix)"""
        
        
#================================================================
     def search_terms(self, terms):
         """ Search for terms in indexed documents.
             Returns: List containing the index of indexed books that
                      contains the query terms.
        """
         docs_indices = []
         for term_index, term in enumerate(terms):
              
              # keep only docs that contains all terms
              if term not in self.term_index:
                 docs_indices = []
                 break 
              
              docs_with_term = self.term_index[term]
              
              if term_index == 0:
                  docs_indices = docs_with_term
              else:
                  # compute intersection between results to remove duplicates
                  docs_indices = set(docs_indices) & set(docs_with_term)
                              
             
         return list(docs_indices)
#===============================================================
     def compute_rank(self, doc_index, terms):
        """Compute tf-idf score of an indexed document.
        Returns: tf-idf of document identified by its index.
        """
        score = 0
        for term in terms:
            term_idx = self.vocabulary[term]
            score += self.tf_idf_matrix[doc_index, term_idx]
        
        return score
#=================================================================    
     def search_corpus(self, query, n_results=10):
        """Return List of search results including the indexed
            object and its respective tf-idf score.
           Assumptions:
           * All terms in the provided query have to be found.
             Otherwise, an empty list will be returned."""
        terms = query.lower().split()
        docs_indices = self.search_terms(terms)
        
        search_results = []
        
        # get the rank and append to search_results
        for i in docs_indices:
            b = self.Books[i]
            b.score = self.compute_rank(i, terms)
            search_results.append(b)

        # sort results from highest score to lowest
        search_results.sort(key=lambda x: x.score, reverse=True) 
        return search_results[:n_results] 
#=================================================================                             