import numpy as np
import pandas as pd
import warnings

def seq_to_num(sequence, delimiter=',', dtype=np.float64, target_split=True, 
               pad=True, pad_maxlen=100, pad_adaptive=True, 
               drop_na_inf=True, nbins=1, bins_by='terms', warn='ignore'):
    """
    Split sequence by delimiter and convert to numbers,
    
    TODO:
    [X] Pad each bin adaptively
    [~] Save bin indexes after nan/inf removal (seems to work)
    
    @param delimiter: split each sequence string using this delimiter
    @param dtype: convert each term into number using this data type
    @param pad: add pre-padding if True
    @param pad_adaptive: pad by median of each bin separatedly if True
    @param pad_maxlen: max length of padded sequence. Cut if longer, pad if shorter.
    @param target_split: split sequences into feature array X and target vector y
    @param drop_na_inf: Filter out np.nan/np.inf values if True (dtype is too small to store a number)
    @param bins_by: sort sequences by # of terms if 'terms', otherwise by max value in sequence
    @param nbins: number of bins. If 1, fallback to no binning.
    @param warnings: action to make
    
    @return: np.array if pad=False and target_split=False and nbins=1
             tuple (X, y) if target_split=True and nbins=1
             list of np.arrays if nbins > 1 and target_split=False
             list of (X, y, idx) tuples if nbins > 1 and target_split=True
    """
    warnings.filterwarnings(warn)
    from keras.preprocessing.sequence import pad_sequences
    warnings.resetwarnings()
    num = sequence.str.split(delimiter).map(dtype)
    if nbins > 1:
        split_vals = num.map(lambda x: len(x) if bins_by == 'terms' else max(x)).sort_values()
        bin_size = int(np.ceil(len(split_vals) / nbins))
        bins = []
        for i in range(nbins):
            idx = split_vals.index[i * bin_size: (i + 1) * bin_size]
            subset = num[idx]
            if pad:
                if pad_adaptive:
                    pad_maxlen = int(subset.map(lambda x: len(x)).median() + 2)
                subset = pad_sequences(subset, value=0.0, maxlen=pad_maxlen, dtype=dtype)
                if drop_na_inf:
                    idx_left = ~(np.isnan(subset).any(axis=1) | np.isinf(subset).any(axis=1))
                    subset = subset[idx_left]
                    idx = idx[idx_left]
                if target_split:
                    subset = subset[:, :-1], subset[:, -1]
                    subset = subset[0], subset[1], idx
                else:
                    subset = subset, idx
            else:
                if target_split:
                    subset = subset.map(lambda x: x[:-1]), subset.map(lambda x: x[-1])
                    subset = subset[0], subset[1], idx
            bins.append(subset)
        return bins
    if pad:
        num = pad_sequences(num, value=0, maxlen=pad_maxlen, dtype=dtype)
        if drop_na_inf:
            num = num[~(np.isnan(num).any(axis=1) | np.isinf(num).any(axis=1))]
        if target_split:
            return num[:, :-1], num[:, -1]
        return num
        if drop_na_inf:
            num = num[~(np.isnan(num).any(axis=1) | np.isinf(num).any(axis=1))]
    if target_split:
        X = num.map(lambda x: x[:-1])
        y = num.map(lambda x: x[-1])
        return X, y
    return num

def prep_submit(predictions: pd.Series, filename='submit.csv'):
    """
    Given predictions Series, format it properly for submition
    
    Note:
        Don't convert floats beforehand! We can store them as strings
    """
    predictions.name = 'Last'
    predictions.index.name = 'Id'
    predictions.to_csv(filename, header=True, float_format='%.f')
    

def acc_score(y_true, y_pred):
    """
    Calculate accuracy for the given predictions vector
    """
    cnt_matches = 0
    if len(y_true) == 0:
        return cnt_matches
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            cnt_matches += 1
    return cnt_matches / len(y_true)

def data_split(path_to_data, frac=.7, remove_dup=True, save=True, random_state=42):
    """
    Split dataset into train/test with possible duplicate removal
    """
    df = pd.read_csv(path_to_data, index_col=0)
    if remove_dup:
        df = df.drop_duplicates(subset='Sequence')
    df_train = df.sample(frac=frac, random_state=random_state)
    # select all rows that are not in `df_train`
    df_test = df.loc[~df.index.isin(df_train.index), :]
    # shuffle dataset to randomize id further
    df_test = df_test.sample(frac=1, random_state=random_state)
    if save:
        df_train.to_csv('../data/train.csv', index=True)
        df_test.to_csv('../data/test.csv', index=True)
