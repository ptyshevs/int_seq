import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense


class RNN:
    def __init__(self, weights_file='pre_train/rnn_weights.h5', input_len=69):
        self.input_len = input_len
        self.model = self._build_model()
        self.model.load_weights(weights_file)
        self.params = {'input_len:': self.input_len, 'model': self.model}
        
    
    def _build_model(self):
        model = Sequential()
        model.add(GRU(128, input_shape=(self.input_len, 1)))
        model.add(Dense(1000, activation='softmax'))
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _prep_data(self, data):
        """
        Pad sequences in data and expand dimention of features
        """
        ind = data.index
        data = pad_sequences(data, maxlen=self.input_len, dtype='int32')
        data = np.expand_dims(data, 2)
        return data, ind
    
    def predict(self, data):
        data, ind = self._prep_data(data)
        pred = self.model.predict(data)
        pred = np.argmax(pred, axis=1)
        return [], ind.tolist(), pred
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"
        
    