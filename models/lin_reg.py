import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import tqdm


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


class LinReg:
    def __init__(self, max_prev=40, poly_deg=1, minlen=3, verbose=False):
        self.max_prev = max_prev
        self.poly_deg = poly_deg
        self.minlen = minlen
        self.verbose = verbose
        self._mod = LinearRegression()
        self.params = {'max_prev': self.max_prev, 'poly_deg': self.poly_deg, 'model': self._mod,
                       'verbose': self.verbose}
    
    def predict(self, data):
        sequences = []
        indices = []
        predictions = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else range(len(data))
        for seq, ind in tqdm.tqdm(zip(data, ind_iter)):
            if len(seq) <= self.minlen:
                continue
            pred = self._pred_best_reg(seq)
            if pred is None:
                continue
            sequences.append(seq)
            indices.append(ind)
            predictions.append(pred)
        return sequences, indices, predictions
    
    def _pred_best_reg(self, seq):
        """
        Try to fit linear regression to previous several numbers, recording score and looking for perfect fit
        """
        min_num = min(len(seq) - 1, 1)
        max_num = min(len(seq) - 1, self.max_prev)
        best_acc, best_num_of_points = -1, -1
        for num_of_points in range(min_num, max_num + 1):
            X, y = self._create_data(seq, num_of_points)
            self._mod.fit(X, y)
            pred = self._mod.predict(X).round()
            try:
                acc = acc_score(y.round(), pred)
            except AttributeError:
                acc = acc_score([round(_) for _ in y], pred)
            if acc > best_acc:
                best_acc = acc
                best_num_of_points = num_of_points
            if np.isclose(best_acc, 1):
                break
        if self.verbose:
            print(f"Best acc: {best_acc}, num of points: {best_num_of_points}")
        # predict
        X, y = self._create_data(seq, best_num_of_points)
        self._mod.fit(X, y)
        pred_data = seq[-best_num_of_points:]
        if self.poly_deg > 1:
            pred_data = PolynomialFeatures(self.poly_deg).fit_transform([pred_data])
            pred = self._mod.predict(pred_data)[0]
        else:
            pred = self._mod.predict([pred_data])[0]
        if np.fabs(pred - pred.round()) > .01:  # not exact solution - exit
            return None
        return pred.round()
    
    def _create_data(self, seq, num_of_points):
        X = [seq[i: i + num_of_points] for i in range(len(seq) - num_of_points)]
        if self.poly_deg > 1:
            X = PolynomialFeatures(self.poly_deg).fit_transform(X)
        y = seq[num_of_points:]
        return X, y

    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"
