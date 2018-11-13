import pandas as pd
import numpy as np


class LinRecRel:
    def __init__(self, max_order=5, minlen=16, maxlen=-1, start_ind=0, verbose=False):
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
            if np.fabs(a_pred - seq[i + order]) > .1:
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


class LinRecRel2(LinRecRel):
    def __init__(self, max_order=5, minlen=16, max_length=16, slice=15, start_index=3):
        """
        Kolya version, with minimal amount of changes
        """
        self.max_order = max_order
        self.max_length = max_length
        self.slice = slice
        self.minlen=minlen
        self.start_index = start_index
    
    def _create_system(self, sequence, order, start_index_a):
        '''
        :param sequence: list, where type(item)=int 
        :param order: recurrent relation order, int(min=2)
        :param start_index_a: int, form which index start
        :return: a,b (ax=b)
        '''
        # validation
        if len(sequence) < start_index_a + order + order + 1:
            # print("Impossible create system")
            return '-100', '-100'
        # x3=ax0+bx1+c
        index_b = start_index_a + order
        a = list()
        b = [sequence[i] for i in range(index_b, index_b + order + 1)]
        for i in range(start_index_a, start_index_a + order + 1):
            a.append([sequence[item] for item in range(i, i + order)])
        a = np.array(a)
        z = np.ones((order + 1, 1))
        a = np.append(a, z, axis=1)
        b = np.array(b)
        return a, b


    def _check_k_order(self, sequence, order, start_index):
        # create system
        try:
            a, b = self._create_system(sequence, order, start_index)
            if a == '-100':
                return -1, 0
            solution = np.linalg.solve(a, b)
        except np.linalg.linalg.LinAlgError:
            return '000', 0
        except IndexError:
            print('index error')
            return '0000', 0
        # check if solution satisfied all items in sequence
        check = self._check_solution(sequence, solution)
        if check:
            return True, solution
        else:
            return False, '001'


    def _check_solution(self, sequence, solution, start=3):
        n = len(sequence)
        # -1 bcs free coef
        order = len(solution) - 1
        for i in range(start, n - len(solution)):
            # індекс не рахує останній елемент, тобто ми не знаємо останнього елементу
            x = np.array([sequence[j] for j in range(i, i + order)])
            s = np.dot(solution[:order], x) + solution[-1]
            if np.fabs(s - sequence[i + order]) > 0.1:
                return False
        return True


    def _predict_1(self, sequence):
        for i in range(1, 6):
            check, solution = self._check_k_order(sequence, i, 3)

            if check is True:
                # order satisfied
                a = [sequence[j] for j in range(len(sequence) - i, len(sequence))]
                # row['predict'] = np.dot(a, solution[:i]) + solution[-1]
                return np.dot(a, solution[:i]) + solution[-1]
        return '0'


    def _validation(self, row):
        row = row.split(',')
        if len(row) < self.minlen:
            return -1
        for item in row:
            if len(item) > self.max_length:
                return -1

        return [int(item) for item in row][-self.slice:]


    # slice - залишити останні
    def predict(self, data):
        sequences = []
        predicted_values = []
        indices = []
        for index, row in data.iterrows():
            sequence = self._validation(row['Sequence'])
            if sequence == -1:
                continue
            sequence = sequence[:-1]
            pred_val = self._predict_1(sequence)
            if pred_val == '0':
                continue
            # заокруглювати значення
            predicted_values.append(int(round(pred_val)))
            indices.append(index)
            sequences.append(row)
        return sequences, indices, predicted_values
    
    
class NonlinearRecRel(LinRecRel):

    def __init__(self):
        pass

    def _create_system(self, sequence, start_index_a):
        '''
        :param sequence: list, where type(item)=int 
        :param order: recurrent relation order, int(min=2)
        :param start_index_a: int, form which index start
        :return: a,b (ax=b)
        '''
        # validation
        order = 6
        # 6 equations
        if len(sequence) < start_index_a + 6+2:
            srart_index_a=0
#         print("Impossible create system")
#         return '-100', '-100'
        if len(sequence) < start_index_a + 6:
            print("Impossible create system")
            return '-100', '-100'
        # x3=cx0^2+c1x1^2+c2x0x1+c3x0+c4x1+c5
        a = list()
        b = [sequence[i] for i in range(start_index_a + 2, start_index_a + 2 + order)]
        for i in range(start_index_a, start_index_a + order):
            a.append(self._create_nonlinear_polynom_equation(sequence[i], sequence[i + 1]))
        a = np.array(a)
        z = np.ones((6, 1))
        a = np.append(a, z, axis=1)
        b = np.array(b)
        return a, b


    def _create_nonlinear_polynom_equation(self, x1, x2):
        return [x1 ** 2, x2 ** 2, x1 * x2, x1, x2]


    def _predict_1(self, sequence, start_index_a):
        a, b = self._create_system(sequence, start_index_a)
        try:
            solution = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            #print('numpy error')
            return '000'
        if self._check_solution(sequence, solution, start_index_a):
            x = sequence[-6:]
            pred_value = self._calculate_nonlinear_polynom([sequence[-2], sequence[-1]], solution)
            return pred_value
        else:
            return '0'


    def _calculate_nonlinear_polynom(self, x, solution):
        return sum([x[0] ** 2 * solution[0], x[1] ** 2 * solution[1], x[0] * x[1] * solution[2], x[0] * solution[3],
                x[1] * solution[4], solution[5]])


    def predict(self, data, start_index=3, maxlen=15, minlen=10, slice=15, verbose=False):
        predicted_values = []
        indices = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else list(range(len(data)))
        for ind, seq in zip(ind_iter, data):
            if len(seq) < minlen:
                continue
            sequence = seq[-maxlen:] if maxlen != -1 else seq
            pred_val = self._predict_1(sequence, start_index)
            if pred_val == '000' or pred_val == '0':
                continue
            predicted_values.append(np.round(pred_val))
            indices.append(ind)
        return predicted_values, indices


    def _check_solution(self, sequence, solution, start_index_a=3):
        n = len(sequence)
        # -1 bcs free coef
        for i in range(start_index_a, n - 2):
            # індекс не рахує останній елемент, тобто ми не знаємо останнього елементу
            x = np.array([sequence[j] for j in range(i, i + 2)])
            s = round(self._calculate_nonlinear_polynom(x, solution))
            if np.fabs(s - sequence[i + 2]) > 0.1:
                return False
        return True