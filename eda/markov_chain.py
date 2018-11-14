import data_processing as dp
import random


# 1-st order chain
def get_trans_matrix_1(seq):
    n = 10
    trans_matrix = [[0] * n for _ in range(n)]

    for i in range(0, len(seq) - 1):
        trans_matrix[int(seq[i])][int(seq[i+1])] += 1

    for row in trans_matrix:
        occurrences_sum = sum(row)
        if occurrences_sum > 0:
            row[:] = [el / occurrences_sum for el in row]

    return trans_matrix


# 2-nd order chain
def get_trans_matrix_2(seq):
    n = 10
    trans_matrix = [[0] * n for _ in range(n * n)]

    for i in range(0, len(seq) - 2):
        prev_state = int(seq[i]) * 10 + int(seq[i + 1])
        curr_state = int(seq[i + 2])
        trans_matrix[prev_state][curr_state] += 1

    for row in trans_matrix:
        occurrences_sum = sum(row)
        if occurrences_sum > 0:
            row[:] = [el / occurrences_sum for el in row]
            if round(sum(row)) > 1:
                print("error")

    return trans_matrix


# 3-rd order chain
def get_trans_matrix_3(seq):
    n = 10
    trans_matrix = [[0] * n for _ in range(n * n * n)]

    for i in range(0, len(seq) - 3):
        prev_state = int(seq[i]) * 100 + int(seq[i + 1]) * 10 + int(seq[i + 2])
        curr_state = int(seq[i + 3])
        trans_matrix[prev_state][curr_state] += 1

    for row in trans_matrix:
        occurrences_sum = sum(row)
        if occurrences_sum > 0:
            row[:] = [el / occurrences_sum for el in row]
            if round(sum(row)) > 1:
                print("error")

    return trans_matrix


# 4-th order chain
def get_trans_matrix_4(seq):
    n = 10
    trans_matrix = [[0] * n for _ in range(n * n * n * n)]

    for i in range(0, len(seq) - 4):
        prev_state = int(seq[i]) * 1000 + int(seq[i + 1]) * 100
        prev_state += int(seq[i + 2]) * 10 + int(seq[i + 3])
        curr_state = int(seq[i + 4])
        trans_matrix[prev_state][curr_state] += 1

    for row in trans_matrix:
        occurrences_sum = sum(row)
        if occurrences_sum > 0:
            row[:] = [el / occurrences_sum for el in row]
            if round(sum(row)) > 1:
                print("error")

    return trans_matrix


def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i


def markov_chain_1_predict(seq):
    size = len(seq)
    trans_matrix = get_trans_matrix_1(seq)
    state = int(seq[size - 2])
    return weighted_choice(trans_matrix[state])


def markov_chain_2_predict(seq):
    size = len(seq)
    if size <= 2:
        return markov_chain_1_predict(seq)
    trans_matrix = get_trans_matrix_2(seq)
    state = int(seq[size - 3]) * 10 + int(seq[size - 2])
    return weighted_choice(trans_matrix[state])


def markov_chain_3_predict(seq):
    size = len(seq)
    if size <= 2:
        return markov_chain_1_predict(seq)
    elif size <= 3:
        return markov_chain_2_predict(seq)
    trans_matrix = get_trans_matrix_3(seq)
    state = int(seq[size-4]) * 100 + int(seq[size-3]) * 10 + int(seq[size-2])
    return weighted_choice(trans_matrix[state])


def markov_chain_4_predict(seq):
    size = len(seq)
    if size <= 2:
        return markov_chain_1_predict(seq)
    elif size <= 3:
        return markov_chain_2_predict(seq)
    elif size <= 4:
        return markov_chain_3_predict(seq)
    trans_matrix = get_trans_matrix_4(seq)
    state = int(seq[size-5]) * 1000 + int(seq[size-4]) * 100 + int(seq[size-3]) * 10 + int(seq[size-2])
    return weighted_choice(trans_matrix[state])


if __name__ == '__main__':
    path = "D:\\UDataSchool\\Project\\Data Sources\\train.csv"
    one_digit_sequences = dp.get_one_digit_sequences(path)
    print(len(one_digit_sequences))
    # print(one_digit_sequences[2])
    # print("Predicted digit: ", markov_chain_predict(one_digit_sequences[2]))

    # markov_chain_1_predict(one_digit_sequences[0])

    num_of_predicted = 0
    for seq in one_digit_sequences:
        if markov_chain_1_predict(seq) == int(seq[len(seq) - 1]):
            num_of_predicted += 1

    print("Number of correctly predicted digits (1-st order): ", num_of_predicted)
    print(f"Algo 1-st order accuracy = {num_of_predicted / len(one_digit_sequences) * 100:.3f} %")

    counter = 0
    num_of_predicted = 0
    for seq in one_digit_sequences:
        if markov_chain_2_predict(seq) == int(seq[len(seq) - 1]):
            num_of_predicted += 1
        elif markov_chain_2_predict(seq) == -1:
            print(counter)
        counter += 1

    print("Number of correctly predicted digits (2-nd order): ", num_of_predicted)
    print(f"Algo 2-nd order accuracy = {num_of_predicted / len(one_digit_sequences) * 100:.3f} %")

    num_of_predicted = 0
    for seq in one_digit_sequences:
        if markov_chain_3_predict(seq) == int(seq[len(seq) - 1]):
            num_of_predicted += 1

    print("Number of correctly predicted digits (3-rd order): ", num_of_predicted)
    print(f"Algo 3-rd order accuracy = {num_of_predicted / len(one_digit_sequences) * 100:.3f} %")

    """
    num_of_predicted = 0
    for seq in one_digit_sequences:
        if markov_chain_4_predict(seq) == int(seq[len(seq) - 1]):
            num_of_predicted += 1

    print("Number of correctly predicted digits (4-th order): ", num_of_predicted)
    print(f"Algo 4-rd order accuracy = {num_of_predicted / len(one_digit_sequences) * 100:.3f} %")
    """
