import numpy as np
import pandas as pd

class LinRecRel:
    def __init__(self, max_order=5, minlen=10, maxlen=-1, start_ind=0, verbose=False):
        """
        Solver of linear recurrent relations of form a_n = c_{n-1} * a_{n-1} + c_{n-2} * a_{n-2} + ... + c_0
        
        @param max_order: maximum number of previous terms considered.
            (Hyperparameter) Increase in order can possibly result in more sequences solved, but increases
                number of FP drastically.

        @param minlen: minimum length of the sequence to be predicted
            (Hyperparameter) Increase results in more reliable predictions, but less of sequences are solved

        @param maxlen: maximum length of the sequence. -1 if the whole sequence should be considered, otherwise
            only last <maxlen> terms are taken into account

        @param start_ind: start index of the sequence to be considered.
            (Hyperparameter) Decrease in index results in more reliable proposed solution check

        @param verbose: output additional information
        """
        self.max_order = max_order
        self.minlen = minlen
        self.maxlen = maxlen
        self.start_ind = start_ind
        self.verbose = verbose
        self.params = {"max_order": max_order, "minlen": minlen, "maxlen": maxlen,
                       "start_ind": start_ind, "verbose":verbose}
    
    def predict(self, data):
        sequences = []
        predictions = []
        indices = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else list(range(len(data)))
        for ind, seq in zip(ind_iter, data):
            if len(seq) < self.minlen:
                continue
            if self.maxlen != -1:
                seq = seq[-self.maxlen:]
            pred_val = self._pred_seq(seq, self.max_order, self.start_ind, self.verbose)
            if np.isnan(pred_val):
#               if verbose:
#                   print(f"{sequence[-5:]}... has no linear combination")
                continue
            sequences.append(seq)
            indices.append(ind)
            predictions.append(np.round(pred_val))  # round floats here
        return sequences, indices, predictions
    
    def _pred_seq(self, seq, max_order, start_ind, verbose=False):
        """
        Check given sequence relation to be linearly dependant
        on previous terms with some constant coefficients.
        """
        for order in range(1, max_order + 1):
            system = self._create_system(seq, order, start_ind)
            if system is None:  # cannot create SOLE, exiting
                break
            else:
                try:  # solve SOLE
                    a, b = system
                    solution = np.linalg.solve(a, b)
                except (np.linalg.linalg.LinAlgError, IndexError):
                    # singular matrix, ie cannot be solved for coefficients
                    continue
            if self._valid_solution(seq, solution, start_ind):
                # order satisfied
                if verbose:
                    print(f"LinRecRel found {order}-th RR. Coefficients: {solution[:3]}...")
                coefs, constant = solution[:-1], solution[-1]
                return seq[-order:] @ coefs + constant
        return np.nan
    
    def _valid_solution(self, seq, solution, start_ind):
        # -1 bcs free coef
        order = len(solution) - 1
        for i in range(start_ind, len(seq) - len(solution) + 1):
            coefs, constant = solution[:-1], solution[-1]
            a_pred = seq[i: i + order] @ coefs + constant
            if not np.isclose(a_pred, seq[i + order]):
                return False
        return True
    
    def _create_system(self, seq, order, start_ind):
        '''
        :param sequence: list, where type(item)=int 
        :param order: recurrent relation order, int(min=2)
        :param start_index_a: int, form which index start
        :return: a,b (ax=b)
        '''
        # validation
        if len(seq) < start_ind + order + order + 1:
            # print("Impossible create system")
            return None
        # x3=ax0+bx1+c
        index_b = start_ind + order
        a = np.array([[seq[item] for item in range(i, i + order)] + [1] for i in range(start_ind + order + 1)])
        b = np.array([seq[i] for i in range(index_b, index_b + order + 1)])
        return a, b
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"