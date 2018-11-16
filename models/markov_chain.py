import pandas as pd
import numpy as np

class MarkovChain:
    def __init__(self, n_prev=4, verbose=False):
        """
        Markov Chain on single-digit sequences

        Note:
            Works on single-digit sequences
        """
        self.n_prev = n_prev
        self.verbose = verbose
        self.params = {'n_prev': self.n_prev, 'verbose': self.verbose}
    
    def predict(self, data):
        sequences = []
        indices = []
        predictions = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else range(len(data))
        for seq, ind in zip(data, ind_iter):
            pred = self._pred(seq)
            if pred == -1:
                continue
            sequences.append(seq)
            indices.append(ind)
            predictions.append(pred)
        return sequences, indices, predictions
    
    def _pred(self, seq):
        size = len(seq)
        ord = self.n_prev
        if size <= 2:
            ord = 1
        elif size <= 3:
            ord = 2
        elif size <= 4:
            ord = 3
        trans_matrix = self._transition_matrix(seq, ord)
        state = sum([int(seq[-_]) * 10 ** (_ - 1) for _ in range(ord, 0, -1)])
        return self._weighted_choice(trans_matrix[state])
    
    def _weighted_choice(self, weights):
        totals = []
        running_total = 0

        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = np.random.random() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i
        # What should we do in this case?
        if self.verbose:
            print(f"No transition from this state exists")
        return -1
    
    def _transition_matrix(self, seq, ord):
        """
        Generate transition matrix of given order
        """
        n = 10
        trans_matrix = [[0] * n for _ in range(n ** ord)]
        for i in range(len(seq) - ord):
            prev_state = sum([int(seq[i + _]) * 10 ** (ord - (_ + 1)) for _ in range(ord)])
            cur_state = int(seq[i + ord])
            trans_matrix[prev_state][cur_state] += 1
    
        for row in trans_matrix:
            oc_sum = sum(row)
            if oc_sum > 0:
                row[:] = [el / oc_sum for el in row]
        return trans_matrix
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"