"""
Class to retrieve list of neighbors from embedding (can be counterfitted) and filtering
 i.e. by discarding words that semantically different, like words for verbs etc.
"""
import numpy as np
import operator
from collections import defaultdict
import pickle
from numpy import linalg as LA
import time


class Neighbors(object):
    def __init__(self, 
                 embedding,
                 in_memory=False,
                 norm=np.linalg.norm,
                 neighbors_filters=[]):
        """
        Input:
            embedding: string
                path to the embedding
            in_memory: boolean
                optional: 
            norm: function
                optional: distance function with just an input (the distance vector)
            neighbors_filters: list
                optional: list of functions to filter the results: each function must recieve as input a list of words, 
                    and outputs a filtered list (for example discarding words that are tagged differently by nltk.pos_tag function)
        """
        self.embedding = embedding
        self.in_memory = in_memory
        self.norm = norm
        self.neighbors_filters = neighbors_filters
        self.word2index, self.index2word, self.index2embedding = self.load_embedding(self.embedding)
        # parameters for the in-memory initialization
        self.words_matrix = None  # vector with all the datapoints
        # keep in memory the entire words' space matrix so operations can be vectorized
        if in_memory is True:
            # this code works in theory and can speedup the process of finding a neighbor,
            #  but the matrix that is stored in memory is big, shape>=(5, 100k)
            self.words_matrix = np.vstack([v for v in self.index2embedding])


    def set_embedding(self, embedding):
        """
        Change on the fly the embedding used to find the neighbors
        """
        self.embedding = embedding
        self.word2index, self.index2word, self.index2embedding = self.load_embedding(embedding)
        print("[logger]: Embedding has been changed successfully to {}".format(self.embedding))

    
    def nearest_neighbors(self, 
                          word, 
                          eps, 
                          k=1,
                          dist=np.linalg.norm):
        """
        Return the k nearest neighbors in the neighborhood of an embedded word
        TODO: complete the documentation
        TODO: implement missing feature for in memory search

        Parameters
        -------
        word: string
            input word
        eps: float
            radius (in the same norm of the embedding) of the n-ball where neighbors are gathered
        k: integer
            optional: maximum number of neighbors to return, regardless of the numbers in the eps-radius
        dist: function
            optional: distance measure between two embeddings

        Returns
        -------
            list of words in the neighborhood, empty if no word is close enough
        """
        neighbors = []
        if self.in_memory is False:
            self.words = [v for v in self.index2word.values()]
            w1 = self.index2embedding[self.word2index[word]]
            for w in self.words:
                w2 = self.index2embedding[self.word2index[w]]
                dist = self.norm(w1-w2)
                if dist <= eps:
                    neighbors.append([w, dist])
            if len(neighbors) == 0:
                print("[logger]: No neighbors in the radius {}".format(eps))
                return [] 
            neighbors = sorted(neighbors, key = operator.itemgetter(1, 0))
            neighbors = [[n[0], n[1]] for n in neighbors[1:k+1]]
        else:
            # this code works and can speedup the process of finding a neighbor,
            #  still the matrix that is stored in memory is big
            indices = operator.itemgetter(*word)(self.word2index)
            words_vectors = np.vstack(operator.itemgetter(*indices)(self.index2embedding))            
            distances = self.norm(self.words_matrix[:,np.newaxis,:].repeat(len(words_vectors), 1) - words_vectors, axis=2)  # exploit numpy vectorization
            argsort_ = distances.argsort(axis=0)[1:k+1]
            neighbors = {}
            for (a,w) in zip(argsort_.T, word):
                for aa in a:
                    if w not in neighbors.keys():
                        try:
                            neighbors[w] = [self.index2word[aa]]
                        except KeyError:
                            print("[logger-WARNING]: index {} for word {} is not in index2word".format(aa, w))
                            neighbors[w] = [self.index2word[2]] # append unknown
                    else:
                        try:
                            neighbors[w].append(self.index2word[aa])
                        except KeyError:
                            print("[logger-WARNING]: index {} for word {} is not in index2word".format(aa, w))
                            neighbors[w].append(self.index2word[2])  # append unknown
        return neighbors

    def filter(self, w, candidates, granularity):
        filtered_candidates = []
        for f in self.neighbors_filters:
            filtered_candidates = f(w, candidates, granularity)
        return filtered_candidates

    def index_to_word(self, word2index) :
        index2word = {value:key for key,value in word2index.items()}
        index2word[0] = '<PAD>'
        index2word[1] = '<START>'
        index2word[2] = '<UNK>'
        return index2word

    def load_embedding(self, emb):
        '''
            Load word embeddings from file.
            Based on https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer/blob/master/GloVe-as-TensorFlow-Embedding-Tutorial.ipynb
        '''
        word_to_index_dict = dict()
        index_to_embedding_array = []
        
        with open(emb, 'r', encoding="utf-8") as emb_file:
            for (i, line) in enumerate(emb_file):
                split = line.split(' ')
                
                word = split[0]
                
                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )
                # use +3 because actual word indexes start at 3 while indexes 0,1,2 are for
                # <PAD>, <START>, and <UNK>
                word_to_index_dict[word] = i+3
                index_to_embedding_array.append(representation)

        _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
        _PAD = 0
        _START = 1
        _UNK = 2
        word_to_index_dict['<PAD>'] = 0
        word_to_index_dict['<UNK>'] = 2
        word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
        index_to_word_dict = self.index_to_word(word_to_index_dict)
        # three 0 vectors for <PAD>, <START> and <UNK>
        index_to_embedding_array = np.array(3*[_WORD_NOT_FOUND] + index_to_embedding_array )
        return word_to_index_dict, index_to_word_dict, index_to_embedding_array

