import numpy as np
import pandas as pd

class Pipeline:
    def __init__(self, models, verbose=False):
        """
        Pipeline is a collection of algorithms used sequentially to predict target.
        Every algorithm (except for fallback) should return tuple of (sequences_solved, indices, prediction)
        
        Last model in a list is considered fallback-prediction.
        @param models: list of (name, model) pairs
        """
        self.models = models
        self.verbose = verbose
        self.stat = None
    
    def summary(self):
        maxnamelen = max([len(name) for name, models in self.models])
        for name, model in self.models:
            print(name.ljust(maxnamelen + 2), model)
    
    def predict(self, data):
        self.stat = []
        predictions = pd.Series(np.zeros(len(data)), index=data.index)
        indices_solved = []
        for name, model in self.models[:-1]:
            _, ind, pred = model.predict(data[~data.index.isin(indices_solved)])
            if self.verbose:
                print(f"solved by {name}: {len(ind)}")
            predictions[ind] = pred
            indices_solved += ind
            self.stat.append((name, len(pred)))
        unsolved_seq = data[~data.index.isin(indices_solved)] 
        fallback_name, fallback_model = self.models[-1]
        if self.verbose:
            print(f"solved by fallback-model {fallback_name}: {len(unsolved_seq)}")
        self.stat.append((fallback_name, len(unsolved_seq)))
        predictions[~predictions.index.isin(indices_solved)] = fallback_model.predict(unsolved_seq)
        return predictions
    
    @property
    def stat_(self):
        """
        Display statistics after prediction was made
        """
        if self.stat is None:
            raise ValueError("Use .predict method to collect statistics")
        for name, nsolved in self.stat:
            print(f"solved by {name.ljust(6)}: {str(nsolved).rjust(8)}")