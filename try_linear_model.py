import pandas as pd
import numpy as np
import math
import time

t = time.time()


def create_system(sequence, order, start_index_a):
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


def check_k_order(sequence, order, start_index):
    # create system
    try:
        a, b = create_system(sequence, order, start_index)
        if a == '-100':
            return -1, 0
        solution = np.linalg.solve(a, b)
    except np.linalg.linalg.LinAlgError:
        return '000', 0
    except IndexError:
        print('index error')
        return '0000', 0
    # check if solution satisfied all items in sequence
    check = check_solution(sequence, solution)
    if check:
        return True, solution
    else:
        return False, '001'


def check_solution(sequence, solution, start=3):
    n = len(sequence)
    # -1 bcs free coef
    order = len(solution) - 1
    for i in range(start, n - len(solution)):
        # індекс не рахує останній елемент, тобто ми не знаємо останнього елементу
        x = np.array([sequence[j] for j in range(i, i + order)])
        s = np.dot(solution[:order], x) + solution[-1]
        if math.fabs(s - sequence[i + order]) > 0.001:
            return False
    return True


def predict_1(sequence):
    for i in range(1, 6):
        check, solution = check_k_order(sequence, i, 3)

        if check is True:
            # order satisfied
            a = [sequence[j] for j in range(len(sequence) - i, len(sequence))]
            # row['predict'] = np.dot(a, solution[:i]) + solution[-1]
            return np.dot(a, solution[:i]) + solution[-1]
    return '0'


def validation(row, max_length=6, slice=15, min_length=8):
    row = row.split(',')
    if len(row) < min_length:
        return -1
    for item in row:
        if len(item) > max_length:
            return -1

    return [int(item) for item in row][-slice:]


# slice - залишити останні
def make_prediction(train, y, max_order=5, max_length=6, slice=15):
    predicted_values = list()
    for index, row in train.iterrows():
        if index == 18:
            rr = 0
        target = y[index]
        sequence = validation(row['Sequence'])
        if sequence == -1:
            predicted_values.append('Bad sequence')
            continue
        sequence = sequence[:-1]
        pred_val = predict_1(sequence)
        if pred_val == '0':
            predicted_values.append("No linear combo")
            continue
        # заокруглювати значення
        predicted_values.append(int(round(pred_val)))
        # predicted_values.append(pred_val)
        # print(pred_val)
    return predicted_values


def get_y(train):
    y = list()
    for index, row in train.iterrows():
        sequence = row['Sequence'].split(',')
        sequence = [int(item) for item in sequence]
        y.append(sequence[-1])
    return y


def score(true, predicted):
    small_error = 0
    n = len(true)
    false_indexes = list()
    count = 0
    for i in range(len(true)):
        if predicted[i] == 'Bad sequence':
            n -= 1
            false_indexes.append(i)
            continue
        elif predicted[i] == 'No linear combo':
            false_indexes.append(i)
            continue
        else:
            error = math.fabs(true[i] - predicted[i])

            if error < 0.001:
                count += 1
                print(i)
                # print(error)
            elif error < 1:
                # print(error)
                small_error += 1
            else:
                false_indexes.append(i)
    print(count)
    print(small_error)
    return count / len(true) * 100, false_indexes


train = pd.read_csv('train.csv')
y = get_y(train)
pred_values = make_prediction(train, y)
score, false_indexes = score(y, pred_values)
print(t - time.time())
data=pd.DataFrame(data=false_indexes, columns=['Bad_indexes'])
data.to_csv('Unrecognised sequences.csv')