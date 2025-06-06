import numpy as np
import pandas as pd
import tqdm

class DiffTable:
    def __init__(self, maxstep=10, stoplen=2, verbose=False):
        """
        Solver based on discrete derivative approximation.
        """
        self.maxstep = maxstep
        self.stoplen=stoplen
        self.verbose = verbose
        self.params = {"maxstep": maxstep, "stoplen": stoplen, "verbose": verbose}
        
    def predict(self, data):
        """
        Calculate next term if first difference is constant
        
        Note:
            Sequences should not be padded.
    
        @param maxstep: maximum step size. All steps in range [1, maxstep] are performed
                        on the first difference. Higher order differences are calculated
                        with step size 1.
        @param stoplen: difference table is stopped when length of seq or its differences
                        of some order is smaller then <stoplen>. 
                        (Hyperparameter) Larger <stoplen> results in more reliable predictions,
                        but shrinks the number of sequences solved. Smaller <stoplen> results
                        in larger number of FP.
    
        @returns:
            * list of sequences that have constant difference
            * list of the corresponding indices
            * list of predicted terms
        """
        sequences = []
        indices = []
        predictions = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else range(len(data))
        for seq, idx in tqdm.tqdm(zip(data, ind_iter)):
            solution_found = False
            for step in range(1, self.maxstep + 1):
                if len(seq) < (step + self.stoplen):
#                 if verbose:
#                     print("Sequence is too small to calculate terms differences:", seq)
                    continue
                last_elems = [seq[-step]]  # last elements of the corresponding differences
                diffs = [_ for _ in seq]
                for i in range(1, (len(seq) - 2) // (step + 1) + 2):
                    diffs = [next - cur for cur, next in zip(diffs, diffs[step if i == 1 else 1:])]
                    if len(diffs) < self.stoplen:  # don't consider diffs that are too short to be reliable
                        break
                    last_elems.append(diffs[-1])
                    uniques = np.unique(diffs)
                    if len(uniques) == 1:
                        if self.verbose:
                            print(f"Seq {seq[:5]}... has constant {i}-th difference {uniques[-1]} with step {step}")
                        sequences.append(seq)
                        indices.append(idx)
                        predictions.append(sum(last_elems))
                        solution_found = True
                        break
                if solution_found:
                    break
        return sequences, indices, predictions
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"
