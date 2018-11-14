import numpy as np
import pandas as pd
import tools


class NonlinearCubeRecRel:
    def __init__(self, slice=15, start_index_a=3, minlen=15):
        self.slice = slice
        self.minlen = minlen
        self.start_index_a = start_index_a

    def _create_system(self, sequence):
        '''
        :param sequence: list, where type(item)=int 
        :param order: recurrent relation order, int(min=2)
        :param start_index_a: int, form which index start
        :return: a,b (ax=b)
        '''
        # x4=x1^3+x2^3+x3^3
        order = 4
        # 4 equations
        if len(sequence) < self.start_index_a + 3 + 2:
            self.srart_index_a = 0
        if len(sequence) < self.start_index_a + 6:
            print("Impossible create system")
            return '-100', '-100'
        # x3=cx0^2+c1x1^2+c2x0x1+c3x0+c4x1+c5
        a = list()
        b = [sequence[i] for i in range(self.start_index_a + 4, self.start_index_a + 4 + order)]
        for i in range(self.start_index_a, self.start_index_a + order):
            a.append(self.create_nonlinear_polynom_equation(sequence[i], sequence[i + 1], sequence[i + 2]))
        a = np.array(a)
        z = np.ones((4, 1))
        a = np.append(a, z, axis=1)
        b = np.array(b)
        return a, b

    def create_nonlinear_polynom_equation(self, x1, x2, x3):
        return [x1 ** 3, x2 ** 3, x3 ** 3]

    def _predict_1(self, sequence):
        a, b = self._create_system(sequence)
        try:
            solution = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            # print('numpy error')
            return '000'
        if self._check_solution(sequence, solution):
            pred_value = self._calculate_nonlinear_polynom([sequence[-3], sequence[-2], sequence[-1]], solution)
            return pred_value
        else:
            return '0'

    def _calculate_nonlinear_polynom(self, x, solution):
        return sum([x[0] ** 3 * solution[0], x[1] ** 3 * solution[1], x[2] ** 3 * solution[2], solution[3]])

    def predict(self, data):
        predicted_values = []
        indices = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else list(range(len(data)))
        for ind, seq in zip(ind_iter, data):
            if len(seq) < self.minlen:
                continue
            sequence = seq[-self.slice:] if self.slice != -1 else seq
            pred_val = self._predict_1(sequence)
            if pred_val == '000' or pred_val == '0':
                continue
            predicted_values.append(np.round(pred_val))
            indices.append(ind)
        return predicted_values, indices

    def _check_solution(self, sequence, solution):
        n = len(sequence)
        # -1 bcs free coef
        for i in range(self.start_index_a, n - 3):
            # індекс не рахує останній елемент, тобто ми не знаємо останнього елементу
            x = np.array([sequence[j] for j in range(i, i + 3)])
            s = round(self._calculate_nonlinear_polynom(x, solution))
            if np.fabs(s - sequence[i + 3]) > 0.1:
                return False
        return True


df_train = pd.read_csv('train.csv', index_col=0)
train_X, train_y = tools.seq_to_num(df_train.Sequence, pad=False)
lin_model = NonlinearCubeRecRel()
pred, ind = lin_model.predict(train_X)
print(len(ind))
score = tools.acc_score(pred, train_y[ind])
print(score)
