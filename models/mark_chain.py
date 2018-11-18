import numpy as np
import pandas as pd
import random


class MarkovChains:
    def __init__(self, max_unique_count, minlen, k, slice=-1):
        # till now optimal par are: max_unique_count=6,minlen=45,slice=-1
        self.max_unique_count = max_unique_count
        self.minlen = minlen
        self.slice = slice
        self.k = k
        self.bad_model = False
        if self.k >= self.minlen:
            self.bad_model = True


    # def create_transition_matrix(self, sequence):
    #     table = dict()
    #     unique = set(sequence)
    #     # create rows
    #     for item in unique:
    #         table[item] = dict()
    #
    #     n = len(sequence)
    #     for item in unique:
    #         d = dict()
    #         for el in unique:
    #             d[el] = 0
    #         for i in range(0, n - 1):
    #             key = sequence[i]
    #             if sequence[i] == item:
    #                 d[sequence[i + 1]] += 1
    #         table[item] = d
    #     return table

    def create_transition_matrix(self, sequence, k):
        # create

        n = len(sequence)
        new_sequence = list()
        seq = [int(item) for item in sequence]
        seq = [str(item) for item in seq]
        for i in range(n - k + 1):
            new_sequence.append(seq[i:i + k])
        unique = list()
        for item in new_sequence:
            if item not in unique:
                unique.append(item)
        matrix = dict()
        for item in unique:
            matrix[str(item)] = 0
        for item in unique:
            d = dict()
            for el in set(sequence):
                d[str(int(el))] = 0
            for i in range(len(seq) - k):
                if seq[i:i + k] == item:
                    d[seq[i + k]] += 1
            matrix[str(item)] = d
        return matrix

    def predict_1(self, sequence):
        predictions = list()
        confidence = list()
        order = list()
        possible = False
        for i in range(self.k):
            # find frequencies of last_elem
            transpose_matrix = self.create_transition_matrix(sequence, i + 1)
            freq_distr = self.create_frequency_distribution(transpose_matrix, sequence[-i - 1:])
            if freq_distr == '000' or freq_distr == '-111':
                continue
            possible = True
            random_value = random.random()
            pred_val, frequency = self.pick_number(freq_distr, random_value)
            predictions.append(pred_val)
            confidence.append(frequency)
            order.append(i + 1)
        # method for weighted prediction
        if possible is False:
            return '000'
        max_freq = confidence.index(max(confidence))
        pred_value = self.weighted_choose(confidence, predictions)
        return pred_value

    def weighted_choose(self, frequencies, predictions, method=1):
        if method == 1:
            # hard voting - choose the that appeared the most frequently
            return max(set(predictions), key=predictions.count)
        elif method == 2:
            # think about metrics, maybe some threshold
            pass

    def create_frequency_distribution(self, transpose_matrix, el):
        # freq_distr -  dict (key=item, value: how often [0,1] appeared after)
        el = [str(int(i)) for i in el]
        freq = transpose_matrix[str(el)]
        if list(freq.values()) == [0] * len(freq.values()):
            return '-111'
        freq_distr = dict()
        s = sum(freq.values())
        for key, value in freq.items():
            try:
                freq_distr[key] = value / s
            except ZeroDivisionError:
                print('zero division')
                return '000'
        return freq_distr

    def pick_number(self, freq_distr, random_value, confidence=0):
        '''
        функція розподілу: генеруємо величину на [0,1). Якщо вона
        є [0,0.4) - 1, [0.4,0.7) - 2, [0.7,0.9) - 3,[0.9,1) - 4.
        :param random_value: величина на [0,1)
        :return: цифра від 1 до 4
        '''
        # confidence is difference between two most frequent elements

        for key, value in freq_distr.items():
            if random_value < value:
                if confidence == 0:
                    return key, value
                elif confidence == 1:
                    # conf = value - sorted(freq_distr, key=freq_distr.values())[1]
                    # return key, conf
                    pass
                else:
                    return '-11111'
            random_value -= value

    def predict(self, data):
        # data is return from seq_to_num( float)
        predicted_values = []
        sequences = []
        indices = []
        ind_iter = data.index if isinstance(data, (np.ndarray, pd.Series)) else list(range(len(data)))
        for ind, seq in zip(ind_iter, data):
            if len(seq) < self.minlen:
                continue
            unique_count = len(set(seq))
            if unique_count > self.max_unique_count:
                continue
            sequence = seq[-self.slice:] if self.slice != -1 else seq

            pred_val = self.predict_1(sequence)
            if pred_val == '000':
                # last ele in train has never appeared before in seq
                continue
            predicted_values.append(float(pred_val))
            indices.append(ind)
            sequences.append(seq)
        return sequences, indices, predicted_values
