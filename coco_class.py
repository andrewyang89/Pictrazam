
import json
import numpy as np
import re
import string
from collections import defaultdict, Counter
from gensim.models.keyedvectors import KeyedVectors


class Coco:
    def __init__(self):
        """
        Contains dictionaries that convert between properties of the dataset.
        
        Dictionaries:
        ------------
        imageID_to_captions: input image ID and get all associated captions
        
        imageID_to_url: input image ID and output URL
        
        imageID_to_captionID: input image ID to output list of associated caption IDs
        
        captionID_to_caption: input caption ID to ouput associated caption
        
        captionID_to_embedding: input caption ID to ouput associated embedding
        
        
        Properties:
        -----------
        image_ids: all image IDs
        
        caption_ids: all caption IDs
        
        captions: all captions
        
        urls: list of all URLs
        
        """
        self.glove = KeyedVectors.load_word2vec_format('glove.6B.50d.txt.w2v', binary=False)
        print ("glove loaded")
        
        with open('captions_train2014.json') as json_file:
            data = json.load(json_file)
            
        self.imageID_to_captions = defaultdict(list)
        self.imageID_to_captionID = defaultdict(list)
        
        # iterating through images
        
        self.urls = []
        self.image_ids = []  
        self.imageID_to_url = {}
        
        for i in range(len(data["images"])):
            self.urls.append(data["images"][i]["coco_url"])
            self.image_ids.append(data["images"][i]["id"])
            self.imageID_to_url[self.image_ids[i]] = self.urls[i]
            
        # iterating through captions
            
        self.caption_ids = []
        self.captions = []
        self.image_ids_repetitions = [] 
            
        
        self.captionID_to_caption = {}
        self.captionID_to_embedding = {}

        for a in range(len(data["annotations"])):
            self.caption_ids.append(data["annotations"][a]["id"])
            self.captions.append(data["annotations"][a]["caption"])
            self.image_ids_repetitions.append(data["annotations"][a]["image_id"])
                
            self.captionID_to_caption[self.caption_ids[a]] = self.captions[a]
            self.imageID_to_captions[self.image_ids_repetitions[a]].append(self.captions[a])        
            self.imageID_to_captionID[self.image_ids_repetitions[a]].append(self.caption_ids[a])
        
        self.caption_embeddings = self.embed_caption(self.captions)
        
        for a in range(len(data["annotations"])):
            self.captionID_to_embedding[self.caption_ids[a]] = self.caption_embeddings[a]
        
        

        
    def embed_caption(self, captions):
        print("began")
        all_counters = [self.to_counter(i) for i in captions]
    
        all_vocab = self.to_vocab(all_counters)

        all_idf = self.to_idf(all_vocab, all_counters)

        all_weights = np.zeros((len(captions), 50))
        print("starting")

        for ind, caption in enumerate(captions): 
            words = self.divide_string(caption)
            N = len(words)
            idf_values = np.zeros((N))


            for j in range(len(words)):
                idx = all_vocab.index(words[j])
                idf_values[j] = all_idf[idx]


            words_glove = np.zeros((N, 50))
            for j in range(len(words)):
                #print(words[j])
                words_glove[j] = self.return_glove(words[j])

            final_values = words_glove

            for j in range(N):
                final_values[j] *= idf_values[j]

            W = np.sum(final_values, axis = 0)

            all_weights[ind] = W


        return all_weights
    
    def to_counter(self, doc):
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
        return Counter(self.strip_punc(doc).lower().split())
    
    def to_vocab(self, counters):
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
    
    def to_idf(self, vocab, counters):
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
        print('finished')
        nt = np.array(nt, dtype=float)
        return np.log10(N / nt)
    
    def strip_punc(self, corpus):
        punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
        return punc_regex.sub('', corpus)

    def divide_string(self, doc):
        return self.strip_punc(doc).lower().split()
    
    def return_glove(self, word):
        return self.glove[word] if word in self.glove else None
