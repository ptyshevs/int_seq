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
    
    def summary(self):
        maxnamelen = max([len(name) for name, models in self.models])
        for name, model in self.models:
            print(name.ljust(maxnamelen + 2), model)
    
    def predict(self, data):
        predictions = pd.Series(np.zeros(len(data)), index=data.index)
        indices_solved = []
        for name, model in self.models[:-1]:
            _, ind, pred = model.predict(data[~data.index.isin(indices_solved)])
            if self.verbose:
                print(f"solved by {name}: {len(ind)}")
            predictions[ind] = pred
            indices_solved += ind
        unsolved_seq = data[~data.index.isin(indices_solved)] 
        fallback_model = self.models[-1][1]
        if self.verbose:
            print(f"solved by fallback-model ({fallback_model}): {len(unsolved_seq)}")
        predictions[~predictions.index.isin(indices_solved)] = fallback_model.predict(unsolved_seq)
        return predictions
        