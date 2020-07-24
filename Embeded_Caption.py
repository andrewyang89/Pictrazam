#!/usr/bin/env python
# coding: utf-8

# In[43]:


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


path = r"./glove.6B.50d.txt.w2v"
t0 = time.time()
glove = KeyedVectors.load_word2vec_format(path, binary=False)
t1 = time.time()
print("elapsed %ss" % (t1 - t0))
# 50d: elapsed 17.67420792579651s
# 100d:


# In[44]:


type(glove['pizza'])


# In[45]:


import re, string
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    return punc_regex.sub('', corpus)


# In[46]:


def divide_string(doc):
    return strip_punc(doc).lower().split()


# In[47]:


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


# In[55]:


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


# In[49]:


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


# In[50]:


def return_glove(word):
    return self.glove[word]


# In[122]:


def se_text(captions):
    """
    Given all the captions. This function will return the Glove embeddings weighted by the IDF for each caption.

    Parameters
    ----------
    captions : Sequence[str]
        An iterable containing a strings that corresponds to captions.

    Returns
    -------

    all_weights : np.ndarray - Shape(N, 50) - where N is number of captions
        This contains each captions with the Glove embeddings weighted by the IDF.
        Each row corresponds to a new caption.
    """

    all_counters = [to_counter(i) for i in captions]

    all_vocab = to_vocab(all_counters)

    all_idf = to_idf(all_vocab, all_counters)

    all_weights = np.zeros((len(captions), 50))


    for ind, caption in enumerate(captions):
        words = divide_string(caption)
        N = len(words)
        idf_values = np.zeros((N))


        for j in range(len(words)):
            idx = all_vocab.index(words[j])
            idf_values[j] = all_idf[idx]


        words_glove = np.zeros((N, 50))
        for j in range(len(words)):
            #print(words[j])
            words_glove[j] = return_glove(words[j])

        final_values = words_glove

        for j in range(N):
            final_values[j] *= idf_values[j]

        W = np.sum(final_values, axis = 0)

        norm = np.linalg.norm(W)
        W /= norm

        all_weights[ind] = W


    return all_weights



# In[ ]:
