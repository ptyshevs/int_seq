import csv

# 8610 one-digit sequences in training set
# 8713 one-digit sequences in test set


def get_one_digit_sequences(path):
    one_digit_sequences = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            is_one_digit = True
            tmp_seq = row[1].split(',')
            for el in tmp_seq:
                if len(el) > 1:
                    is_one_digit = False
                    break
            if is_one_digit:
                one_digit_sequences.append(''.join(row[1].split(',')))
    return one_digit_sequences


"""
def get_eq_len_sequences(path):
    eq_len_sequences = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            is_eq_len = True
            tmp_seq = row[1].split(',')
            el_len = len(tmp_seq[0])
            for el in tmp_seq:
                if len(el) != el_len:
                    is_eq_len = False
                    break
            if is_eq_len:
                eq_len_sequences.append(''.join(tmp_seq))
    return eq_len_sequences
"""

if __name__ == '__main__':
    path = "D:\\UDataSchool\\Project\\Data Sources\\test.csv"
    one_digit_sequences = get_one_digit_sequences(path)
    print(len(one_digit_sequences))
    # eq_len_sequences = get_eq_len_sequences(path)
    # print(len(eq_len_sequences))
