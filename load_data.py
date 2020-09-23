import numpy as np
import re
import tensorflow as tf
import string
import torch
import torch.nn  as nn
from pandas import read_csv
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical  #one-hot encode target column
from torchtext import data
from torchtext import datasets

from glove_utils import load_embedding, pad_sequences, load_embedding_clear

class RNN(nn.Module):
    """
    RNN sample class used for torch models (POPQORN)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, 
                            dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        hidden = self.dropout(hidden[0,:,:])            
        return self.fc(hidden)
    def predict(self, x):
        """
        Same as forward but input is a list that is evaluated element by element
            and th return is converted to a numpy array/tensor
        """
        if not isinstance(x, type([])):
            x = [x]
        numpy_results = np.zeros(shape=(len(x), 7))
        r = torch.Tensor()
        for i in range(len(x)):
            r = self.forward(x[i])
            numpy_results[i] = r.detach().numpy()
        return numpy_results

def load_IMDB_dataset(embedding, emb_dims, maxlen, num_samples=-1, return_text=False):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True)
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    n = num_samples  # -1 means all the data-points
    X_train = X_train[:n]
    X_test = X_test[:n]
    y_train = y_train[:n]
    y_test = y_test[:n]
    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+3) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3
    id_to_word = {value:key for key,value in word_to_id.items()}
    X_train = [[id_to_word[x] for x in xx] for xx in X_train]
    X_test = [[id_to_word[x] for x in xx] for xx in X_test]
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test]) 
    if return_text is True:
        return (X_train, y_train), (X_test, y_test)
    else:
        word2index, _, index2embedding = load_embedding(embedding)
        X_train = [[index2embedding[word2index[x]] for x in xx] for xx in X_train]
        X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
        X_train = np.asarray(pad_sequences(X_train, maxlen=maxlen, emb_size=emb_dims))
        X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
        # reshape inputs
        ksize = int(maxlen**0.5)
        X_train = X_train.reshape(n, ksize, ksize, emb_dims)
        X_test = X_test.reshape(n, ksize, ksize, emb_dims)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        # normalize between -0.5, 0.5 (3.0639 is the absolute max value)
        denominator = 2*np.max(np.abs(index2embedding))
        X_train /= denominator
        X_test /= denominator
        print("[logger]: MIN and MAX of the train-test are, respectively:")
        print("[logger]: (Train,Test)_Min({}, {})".format(np.min(X_train), np.min(X_test)))
        print("[logger]: (Train-Test)_Max({}, {})".format(np.max(X_train), np.max(X_test)))
        return (X_train, y_train), (X_test, y_test)


def load_SST_dataset(embedding, emb_dims, maxlen, num_samples=-1, return_text=False):
    """
    Test all the models specified in global variable models 
     with the respective embedding specified in variable embeddings.
    Returns a vector where each entry contains the accuracy of the model against the full test-set:
     the vector has the same size as the number of models specified in models.
    """
    X = read_csv('./data/datasets/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
    y = []
    for i in range(len(X)):
        r, s = X[i]
        X[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y.append((0 if s.strip()=='negative' else 1))
    X = X[:,0]
    n = -1  # you may want to take just some samples (-1 to take them all)
    X = X[:n]
    y = y[:n]
    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+3) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3
    X = np.array([np.array(x) for x in X]) 
    if return_text is False:
        word2index, _, index2embedding = load_embedding(embedding)
        ksize = int(maxlen**0.5) 
        denominator = 2*np.max(np.abs(index2embedding))
        X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X]
        X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
        X_test = X_test.reshape(len(X_test), ksize, ksize, emb_dims)
        y_test = to_categorical(y, num_classes=10)
        X_test /= denominator
        return (None, None), (X_test, y_test)  # consistent return
    else:
        return (None, None), (X, y)


def load_QA_dataset(embedding, emb_dims, maxlen, num_samples=-1, return_text=False):
    X_test, y_test = [], []
    with open('./data/datasets/QA_dataset/TREC_10.label') as f: 
        for line in f:
            line = re.sub('[!#?,.";`]', '', line.rstrip())
            label, txt = line.split()[0], line.split()[1:]
            y_test.append(label.split(':')[0])
            X_test.append(txt)
    # take just some samples
    n = -1
    X_test = X_test[:n]
    y_test = y_test[:n]
    if return_text is False:
        # Select the embedding
        word2index, _, index2embedding = load_embedding(embedding)
        X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
        X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
        # reshape inputs
        ksize = int(maxlen**0.5)
        X_test = X_test.reshape(n, ksize, ksize, emb_dims)
        # turn labels into numerical categories
        unique_labels = np.array(['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM'])
        for i in range(len(y_test)):
            y_test[i] = np.argwhere(unique_labels==y_test[i])[0,0]
        y_test = to_categorical(y_test, num_classes=10)
        # normalize between -0.5, 0.5
        denominator = 2*np.max(np.abs(index2embedding))
        X_test /= denominator
        return (None, None), (X_test, y_test)  # consistent return
    else:
        return (None, None), (X_test, y_test)

def load_AG_dataset(embedding, emb_dims, maxlen, num_samples=-1, return_text=False):
    X_train = read_csv('./data/datasets/AG_News/train.csv', sep=',',header=None).values
    X_test = read_csv('./data/datasets/AG_News/test.csv', sep=',',header=None).values
    y_train, y_test = [], []
    for i in range(len(X_train)):
        s, t, r = X_train[i]  # score, title, review (comma separated in the original file)
        X_train[i][0] = [w.lower() for w in t.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        X_train[i][0].extend([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
        X_train[i][0] = [x for x in X_train[i][0] if x!='']
        y_train.append(s)
    for i in range(len(X_test)):
        s, t, r = X_test[i]
        X_test[i][0] = [w.lower() for w in t.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        X_test[i][0].extend([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
        X_test[i][0] = [x for x in X_test[i][0] if x!='']
        y_test.append(s)
    X_train, X_test = X_train[:,0], X_test[:,0]
    n = num_samples  # you may want to take just some samples (-1 to take them all)
    X_train = X_train[:n]
    X_test = X_test[:n]
    y_train = y_train[:n]
    y_test = y_test[:n]
    if return_text is False:
        # Select the embedding
        word2index, _, index2embedding = load_embedding(embedding)
        X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
        X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
        # reshape inputs
        ksize = int(maxlen**0.5)
        X_test = X_test.reshape(n, ksize, ksize, emb_dims)
        # turn labels into numerical categories
        unique_labels = np.array(['1', '2', '3', '4'])
        for i in range(len(y_test)):
            y_test[i] = int(y_test[i])
        y_test = to_categorical(y_test, num_classes=10)
        # normalize between -0.5, 0.5
        denominator = 2*np.max(np.abs(index2embedding))
        X_test /= denominator
        return (None, None), (X_test, y_test)  # consistent return
    else:
        return (None, None), (X_test, y_test)

def load_NEWS_dataset(data_prefix, emb_dims, maxlen, num_samples=-1, return_text=False):
    """
    Dataset for POPQORN/Torch evaluation
    """
    dataset_path = './data/datasets/POPQORN_news/{}/{}d/'.format(data_prefix, emb_dims)
    category_list = np.array(["sport", "world", "us", "business", "health", "entertainment", "sci_tech"])
    y_test = np.load(dataset_path + 'test_labels.npy', allow_pickle=True)
    Y = []
    for label in y_test:
        idx = np.argwhere(category_list==label)
        Y.append(idx[0][0])
    y_test = to_categorical([[y] for y in Y], num_classes=7)
    if return_text is False:
        INPUT_DIM = 10002  # copy-pasta     
        model = RNN(INPUT_DIM, emb_dims, 256, 7, 0)
        X_test = np.load(dataset_path + 'test_vectors.npy', allow_pickle=True)
        X_test = [torch.stack([torch.from_numpy(x).to(torch.float32) for x in xx]).unsqueeze(0) for xx in X_test]
        return (None, None), (X_test, y_test), model
    else:
        X_test = np.load(dataset_path + 'test_words.npy', allow_pickle=True)
        X_indices = np.load(dataset_path + 'test_indices.npy', allow_pickle=True)
        return (None, None), (X_test, X_indices)