import pandas as pd
import numpy as np
import math


class LinearModel:
    def __init__(self, max_order=5, start_index_a=3, minlen=16, slice=14):
        self.max_order = max_order
        self.minlen = minlen
        self.slice = slice
        self.start_index_a = start_index_a

    def create_system(self, sequence, order):
        if len(sequence) < self.start_index_a + order + order + 1:
            # print("Impossible create system")
            return '-100', '-100'
        # x3=ax0+bx1+c
        index_b = self.start_index_a + order
        a = list()
        b = [sequence[i] for i in range(index_b, index_b + order + 1)]
        for i in range(self.start_index_a, self.start_index_a + order + 1):
            a.append([sequence[item] for item in range(i, i + order)])
        a = np.array(a)
        z = np.ones((order + 1, 1))
        a = np.append(a, z, axis=1)
        b = np.array(b)
        return a, b

    def check_solution(self, sequence, solution):
        n = len(sequence)
        order = len(solution) - 1
        for i in range(self.start_index_a, n - len(solution)):
            x = np.array([sequence[j] for j in range(i, i + order)])
            s = np.dot(solution[:order], x) + solution[-1]
            if math.fabs(s - sequence[i + order]) > 0.001:
                return False
        return True

    def check_k_order(self, sequence, order):
        # create system
        try:
            a, b = self.create_system(sequence, order)
            if a == '-100':
                return -1, 0
            solution = np.linalg.solve(a, b)
        except np.linalg.linalg.LinAlgError:
            return '000', 0
        except IndexError:
            print('1index error')
            return '0000', 0
        # check if solution satisfied all items in sequence
        check = self.check_solution(sequence, solution)
        if check:
            return True, solution
        else:
            return False, '001'

    def predict_1(self, sequence):
        for i in range(1, self.max_order):
            check, solution = self.check_k_order(sequence, i)
            if check is True:
                # order satisfied
                a = [sequence[j] for j in range(len(sequence) - i, len(sequence))]
                # дивимося як значення відхиляється від цілого найближчого. Якщо відхилення велике, то ми overfit.
                pred_val = np.dot(a, solution[:i]) + solution[-1]
                if math.fabs(pred_val - round(pred_val)) < 0.1:
                    return np.dot(a, solution[:i]) + solution[-1]
        return '0'

    def predict(self, data):
        predicted_values = []
        indices = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else list(range(len(data)))
        for ind, seq in zip(ind_iter, data):
            if len(seq) < self.minlen:
                continue
            sequence = seq[-self.slice:] if self.slice != -1 else seq
            pred_val = self.predict_1(sequence)
            if pred_val == '000' or pred_val == '0':
                continue
            predicted_values.append(np.round(pred_val))
            indices.append(ind)
        return predicted_values, indices
