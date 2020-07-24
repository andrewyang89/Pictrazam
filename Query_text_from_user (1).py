#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import numpy as np
import time
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import re, string
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    return punc_regex.sub('', corpus)


# In[3]:


def divide_string(doc):
    return strip_punc(doc).lower().split()


# In[4]:


def to_counter(doc):
    """ 
    Produce word-count of document, removing all punctuation
    and making all the characters lower-cased.
    
    Parameters
    ----------
    doc : str
    
    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    return Counter(strip_punc(doc).lower().split())


# In[5]:


def to_vocab(counters):
    """ 
    Takes in an iterable of multiple counters, and returns a sorted list of unique words 
    accumulated across all the counters
    
    [word_counter0, word_counter1, ...] -> sorted list of unique words
    
    Parameters
    ----------
    counters : Iterable[collections.Counter]
        An iterable containing {word -> count} counters for respective
        documents.
    
    Returns
    -------
    List[str]
        An alphabetically-sorted list of all of the unique words in `counters`"""
    vocab = set()
    for counter in counters:
        vocab.update(counter)
    return sorted(vocab)


# In[6]:


def to_idf(vocab, counters):
    """ 
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.
    
    Parameters
    ----------
    vocab : Sequence[str]
        Ordered list of words that we care about.

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.
    
    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `vocab`, storing
        the IDF for each term `t`: 
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of 
        documents in which the term `t` occurs.
    """
    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in vocab]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)


# In[7]:


def return_glove(word):
    return glove[word]


# In[10]:


def query_text(all_idf, all_vocab):
    """
    Returns the weighted query text. 
    
    Parameters
    ----------
    
    all_idf: np.ndarray
        This contains all the IDF values for all 
        vocab in the captions.
        
    all_vocab: List
        This contains all the vocab in the captions
        sorted alphabetically.
        
    
    Returns
    -------
    
    weights : np.ndarray - Shape(50,) - 
        This contains text query weighted by GloVe embeddings
        and IDF values. 

    """
    text = input("Enter query text: ")
    
    counter = to_counter(text)
    vocab = to_vocab(counter)

    weights = np.zeros((1, 50))
    
    
    words = divide_string(text)
    N = len(words)
    idf_values = np.zeros((N))


    for j in range(len(words)):
        
        if words[j] in all_vocab:
    
            idx = all_vocab.index(words[j])
        
            idf_values[j] = all_idf[idx]
        else:
            idf_values[j] = 1


    words_glove = np.zeros((N, 50))
    for j in range(len(words)):

        words_glove[j] = return_glove(words[j])

    final_values = words_glove

    for j in range(N):
        final_values[j] *= idf_values[j]

    W = np.sum(final_values, axis = 0)
    
    norm = np.linalg.norm(W)
    W /= norm

    weights = W
    return weights


# In[ ]:




