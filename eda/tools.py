import numpy as np
import pandas as pd


def seq_to_num(sequence, delimiter=',', dtype=np.float64, pad=True, pad_maxlen=100, target_split=True, drop_na_inf=True, bins_by='terms', nbins=1):
    """
    Split sequence by delimiter and convert to numbers,
    
    TODO:
    [ ] Pad each bin separatedly
    [ ] Save bin indexes AFTER THE FUCKING NAN/INF REMOVAL
    
    @param delimiter: split each sequence string using this delimiter
    @param dtype: convert each term into number using this data type
    @param pad: add pre-padding if True
    @param pad_maxlen: max length of padded sequence. Cut if longer, pad if shorter.
    @param target_split: split sequences into feature array X and target vector y
    @param drop_na_inf: Filter out np.nan/np.inf values if True (dtype is too small to store a number)
    @param bins_by: sort sequences by # of terms if 'terms', otherwise by max value in sequence
    @param nbins: number of bins. If 1, fallback to no binning.
    
    @return: np.array if pad=False and target_split=False and nbins=1
             tuple (X, y) if target_split=True and nbins=1
             list of np.arrays if nbins > 1 and target_split=False
             list of (X, y, idx) tuples if nbins > 1 and target_split=True
    """
    from keras.preprocessing.sequence import pad_sequences

    num = sequence.str.split(delimiter).map(dtype)
    if nbins > 1:
        split_vals = num.map(lambda x: len(x) if bins_by == 'terms' else max(x)).sort_values()
        bin_size = int(np.ceil(len(split_vals) / nbins))
        bins = []
        for i in range(nbins):
            idx = split_vals.index[i * bin_size: (i + 1) * bin_size]
            subset = num[idx]
            if pad:
                subset = pad_sequences(subset, value=0.0, maxlen=pad_maxlen, dtype=dtype)
            if drop_na_inf:
                idx_left = ~(np.isnan(subset).any(axis=1) | np.isinf(subset).any(axis=1))
                subset = subset[idx_left]
                idx = idx[idx_left]
            if target_split:
                subset = subset[:, :-1], subset[:, -1]
            subset = subset, idx
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




def acc_score(y_true, y_pred):
    cnt_matches = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            cnt_matches += 1
    return cnt_matches / len(y_true)
