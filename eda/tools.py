import numpy as np

def transform_df(numbers, terms, percentile_cut=95):
    """
    Transform dataset into feature matrix X and target variable y
    @param data: DataFrame of sequence of numbers
    @param percentile_cut: remove outliers above the given percentile
    
    return: 
    """
    if percentile_cut:
        cut = int(np.ceil(np.percentile(terms, percentile_cut)))
        X = numbers.apply(lambda x: x[:cut])
        y = numbers.apply(lambda x: x[min(len(x) - 1, cut)])
    else:
        X = numbers.apply(lambda x: x[:-1])
        y = numbers.apply(lambda x: x[-1])
    return X, y

from keras.preprocessing.sequence import pad_sequences

def truncate_numbers(numbers, maxlen=100, normalize=True):
    """
    Truncate sequences to have max length of <maxlen>
    and split into X, y
    """
    trunc = pad_sequences(numbers, maxlen, dtype=np.float128)
    X = trunc[:, :-1]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = trunc[:, -1]
    return X, y
