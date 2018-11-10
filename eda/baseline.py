import numpy as np

class Baseline:
    def __init__(self, mode='mode'):
        """
        Class used as a baseline for submission, when every other appoach
        has appeared to be unsuccessful.
        
        @param mode: control what algorithm is used for prediction. Possible values:
              * mode: most frequent term in a sequence (default)
              * median: predict median for a sequence
        """
        self.pred_func = self._mode
        if mode == 'median':
            self.pred_func = self._median
    
    def predict(self, data):
        pred = np.zeros_like(data, dtype=np.float64)
        for i, seq in enumerate(data):
            if len(seq) == 0:
                continue
            pred[i] = self.pred_func(seq)
        return pred
    
    def _mode(self, seq):
        """
        Calculate mode for a given sequence
        """
        unique, counts = np.unique(seq, return_counts=True)
        return unique[np.where(counts == max(counts))][0]

    def _median(self, seq):
        return np.median(seq)
    