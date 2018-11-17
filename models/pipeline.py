import numpy as np
import pandas as pd

class Pipeline:
    def __init__(self, models, fallback=None, verbose=False):
        """
        Pipeline is a collection of algorithms used sequentially to predict target.
        Every algorithm (except for fallback) should return tuple of (sequences_solved, indices, prediction)
        
        @param models: list of (name, model, filter) pairs
        @param fallback: model used on sequences that remained unsolved after passing through all models.
        @param verbose: output additional information during predict
        """
        self.models = models
        self.fallback = fallback
        self.verbose = verbose
        self.stat = None
    
    def summary(self):
        maxnamelen = max([len(name) for name, models in self.models])
        for name, model in self.models:
            print(name.ljust(maxnamelen + 2), model)
        if self.fallback:
            print(f"Fallback model: {self.fallback}")
        else:
            print("No fallback model")
    
    def predict(self, data):
        """
        Pass data through the pipeline
        
        @return:
            If fallback model was specified, series of predictions returned for whole
            data. Otherwise, return indices of sequences solved and corresponding predictions.
        """
        self.stat = []
        predictions = pd.Series(np.zeros(len(data)), index=data.index)
        indices_solved = []
        for name, model, filt in self.models:
            subset = data[~data.index.isin(indices_solved)]
            if filt is not None:
                subset = subset[subset.map(filt)]
            _, ind, pred = model.predict(subset)
            if self.verbose:
                print(f"solved by {name}: {len(ind)}")
            predictions[ind] = pred
            indices_solved += ind
            self.stat.append((name, len(pred)))
        if self.fallback is None:  # no fallback model - return indices and predicitons
            predictions = predictions[predictions.index.isin(indices_solved)]
            return indices_solved, predictions
        unsolved_seq = data[~data.index.isin(indices_solved)] 
        if self.verbose:
            print(f"solved by fallback-model {self.fallback}: {len(unsolved_seq)}")
        self.stat.append((str(self.fallback), len(unsolved_seq)))
        predictions[unsolved_seq.index] = self.fallback.predict(unsolved_seq)
        return predictions

    def predict1(self, seq):
        for name, model, filt in self.models:
            data = seq
            if filt is not None:
                if not filt(data):
                    continue
            _, ind, pred = model.predict(pd.Series([data]))
            if len(ind) > 0:
                print(f"predicted by {name}: {pred}")
                return name, pred[0]
        print("fall-backing")
        return 'fall-back model', self.fallback.predict([seq])[0]

    @property
    def stat_(self):
        """
        Display statistics after prediction was made
        """
        if self.stat is None:
            raise ValueError("Use .predict method to collect statistics")
        for name, nsolved in self.stat:
            print(f"solved by {name.ljust(6)}: {str(nsolved).rjust(8)}")