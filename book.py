from collections import defaultdict

class Book:
    
    def __init__(self, bookID, title, author, metadata):
       self.ID = bookID 
       self.title = title
       self.author = author
       self.metadata = metadata
       self.words_count = defaultdict(int) # data dictionary <word: freq. of occurence per book>
       
       # increment words_count using metadata
       for word in metadata.split():
            self.words_count[word] += 1
       self.score = -1  # initial score
       
       
    def get_words_count(self):
        return self.words_count
        
    def set_words_count(self, x):
        self.words_count = x
          
    def count_for_word(self, word):
        """Returns: Number of occurrences of a given word."""
        return self.words_count[word] if word in self.words_count else 0
        
    def __repr__(self):
            return "score: %s, id: %s, title: %s, author: %s" % \
                   (self.score, self.ID, self.title, self.author)
       