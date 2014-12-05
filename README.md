## Introduction

In information retrieval or text mining, the term frequency – inverse document frequency also called tf-idf, is a well know method to evaluate how important is a word in a document. tf-idf are also a very interesting way to convert the textual representation of information into a Vector Space Model (VSM), or into sparse features.
VSM, is a space where text is represented as a vector of numbers instead of its original string textual representation; the VSM represents the features extracted from the document.

## Program Goal

Given a big corpus of book data (about million records): bookid, Author and Title, this program implements a tf*idf index in memory which does the following:
∙ Read English text data into the index
∙ For a given query, output the top 10 results ranked by their tf-idf scores

## In order to run this program:

1. Store the corpus file under the name “title_author.tab.txt” in the folder called “data”.
2. open a terminal and change directory to BookIndexing then type

   ** $ python3 main.py **

## Perquisites:

The following programs should be installed inorder to run the program:

- Python   v. 3.4
- scipy    v. 0.14.0
- numpy    v.1.9.1 

## Reference:

- <http://pyevolve.sourceforge.net/wordpress/?p=1589>
- <http://aimotion.blogspot.ca/2011/12/machine-learning-with-python-meeting-tf.html>
