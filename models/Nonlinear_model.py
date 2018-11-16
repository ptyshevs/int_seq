import pandas as pd
import numpy as np
import math


class NonLinearModel:
    def __init__(self, start_index_a=3, minlen=12, slice=13):
        self.start_index_a = start_index_a
        self.minlen = minlen
        self.slice = slice

    def create_system(self, sequence):
        '''
        :param sequence: list, where type(item)=int 
        :param order: recurrent relation order, int(min=2)
        :param start_index_a: int, form which index start
        :return: a,b (ax=b)
        '''
        # validation
        order = 6
        # 6 equations
        if len(sequence) < self.start_index_a + 6:
            print("Impossible create system")
            return '-100', '-100'
        # x3=cx0^2+c1x1^2+c2x0x1+c3x0+c4x1+c5
        a = list()
        b = [sequence[i] for i in range(self.start_index_a + 2, self.start_index_a + 2 + order)]
        for i in range(self.start_index_a, self.start_index_a + order):
            a.append(self.create_nonlinear_polynom_equation(sequence[i], sequence[i + 1]))
        a = np.array(a)
        z = np.ones((6, 1))
        a = np.append(a, z, axis=1)
        b = np.array(b)
        return a, b

    def predict_1(self, sequence):
        a, b = self.create_system(sequence)
        try:
            solution = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
#             print('numpy error')
            return '000'
        if self.check_solution(sequence, solution):
            pred_value = self.calculate_nonlinear_polynom([sequence[-2], sequence[-1]], solution)
            return pred_value
        else:
            return '0'

    def check_solution(self, sequence, solution):
        n = len(sequence)
        # -1 bcs free coef
        for i in range(self.start_index_a, n - 2):
            # індекс не рахує останній елемент, тобто ми не знаємо останнього елементу
            x = np.array([sequence[j] for j in range(i, i + 2)])
            s = round(self.calculate_nonlinear_polynom(x, solution))
            if math.fabs(s - sequence[i + 2]) > 0.001:
                return False
        return True

    def create_nonlinear_polynom_equation(self, x1, x2):
        return [x1 ** 2, x2 ** 2, x1 * x2, x1, x2]

    def calculate_nonlinear_polynom(self, x, solution):
        return sum([x[0] ** 2 * solution[0], x[1] ** 2 * solution[1], x[0] * x[1] * solution[2], x[0] * solution[3],
                    x[1] * solution[4], solution[5]])

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
        return [], indices, predicted_values
