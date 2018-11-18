import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense


class RNN:
    def __init__(self, model='big', input_len=69, min_val=0, max_val=2000):
        weights_file = 'pre_train/rnn_weights.h5'
        if model == 'big':
            self.input_len = 10
            weights_file = 'pre_train/rnn_10.h5'
        elif model == 'deep':
            self.input_len = 10
            weights_file = 'pre_train/rnn_deep_10.h5'
        else:
            self.input_len = 69
        self.min_val = min_val
        self.max_val = max_val
        self.model = self._build_model(model)
        self.model.load_weights(weights_file)
        self.params = {'input_len:': self.input_len, 'model': self.model}
        
    
    def _build_model(self, model_type):
        model = Sequential()
        if model_type == 'big':
            model.add(LSTM(256, input_shape=(self.input_len, 1), return_sequences=True))
            model.add(LSTM(256))
        elif model_type == 'deep':
            model.add(LSTM(self.input_len, input_shape=(self.input_len, 1), return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len, return_sequences=True))
            model.add(LSTM(self.input_len))
        else:
            model.add(GRU(128, input_shape=(self.input_len, 1)))
        model.add(Dense(self.max_val, activation='softmax'))
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _prep_data(self, data):
        """
        Pad sequences in data and expand dimention of features
        """
        ind = data.index if isinstance(data, (np.ndarray, pd.Series)) else range(len(data))
        data = pad_sequences(data, maxlen=self.input_len, dtype='int32')
        data = np.expand_dims(data, 2)
        return data, ind
    
    def predict(self, data):
        data, ind = self._prep_data(data)
        pred = self.model.predict(data)
        pred = np.argmax(pred, axis=1)
        if hasattr(ind, 'tolist'):
            return [], ind.tolist(), pred
        else:
            return [], list(ind), pred
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"
        
def arithmetic_prog(n_terms, a1=1, d=1):
    a = []
    for i in range(n_terms):
        a.append(a1)
        a1 += d
    return a

def geometric_prog(n_terms, a1=1, r=2):
    a = []
    for i in range(n_terms):
        a.append(a1)
        a1 *= r
    return a

def fibonacci(n_terms, start_ind=1):
    seq = []
    a, b = 1, 1
    for i in range(start_ind):
        a, b = b, a + b
    for i in range(n_terms):
        seq.append(a)
        a, b = b, a + b
    return seq

# unused -- grows too fast
def factorial(n_terms):
    prod = 1
    seq = []
    for i in range(1, n_terms + 1):
        prod *= i
        seq.append(prod)
    return seq

    

class RNNData:
    def __init__(self, seqlen, aug_frac=0, minval=0, maxval=2000):
        self.seqlen = seqlen
        self.aug_frac = aug_frac
        self.data_filt = lambda seq: len(seq) > 2 and np.all([minval <= x < maxval for x in seq])
        self.val_filt = lambda x: minval <= x < maxval
    
    def transform(self, data):
        """
        Pass data from seq_to_num without padding
        """
        data = data[data.map(self.data_filt)]
        X, y = [], []
        for seq in data:
            if len(seq) <= self.seqlen:
                X += [list(map(int, [0] * (self.seqlen - (len(seq) - 1)) + seq[:-1].tolist()))]
                y += [int(seq[-1])]
                continue
            x1 = [seq[i: i + self.seqlen] for i in range(len(seq) - self.seqlen)]
            y1 = list(map(int, seq[self.seqlen:].tolist()))
            X += x1
            y += y1
        X = np.array(X)
        X = np.expand_dims(X, 2)
        y = np.array(y)
        y = np.expand_dims(y, 1)
        return (X, y) if self.aug_frac == 0 else self.augment_data(X, y)
    
    def augment_data(self, X=None, y=None, n_samples=None):
        """
        Fill train dataset with generated samples from various common sequences
        
        @param n_samples: if not None, only artificial samples are returned
        """
        if self.aug_frac == 0 and n_samples is None:
            return X, y
        only_aug = True
        if n_samples is None:
            only_aug = False
            n_samples = int(len(y) * self.aug_frac)
        aug_X = np.zeros((n_samples, self.seqlen))
        aug_y = np.zeros((n_samples, 1))
        for i in np.arange(n_samples):
            aug_X[i], aug_y[i] = self._aug_dispatch()
        if only_aug:
            return np.expand_dims(aug_X, 2), aug_y
        X = np.append(X, np.expand_dims(aug_X, 2), axis=0)
        y = np.append(y, aug_y, axis=0)
        return X, y
    
    def _aug_dispatch(self, largest_start=10):
        """
        Given length of the sequence, generate sample from common sequences
        """
        a1 = np.random.randint(1, largest_start)
        d = np.random.randint(1, largest_start)
        choice = np.random.randint(0, 3)
        if choice == 0:
            seq = arithmetic_prog(self.seqlen + 1, a1, d)
        elif choice == 1:
            seq = geometric_prog(self.seqlen + 1, a1, d)
        elif choice == 2:
            seq = fibonacci(self.seqlen + 1, a1)
        seq = list(filter(self.val_filt, seq))
        if len(seq) < (self.seqlen + 1):
            seq = [0] * ((self.seqlen + 1) - len(seq)) + seq
        return seq[:-1], seq[-1]
    