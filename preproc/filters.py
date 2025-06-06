import numpy as np

# Wrapper around target variable
target_wrap = lambda val, f: f([val])

# Filter out sequences that consist of single digit (MChain)
markov_filter = lambda seq: len(seq) > 0 and np.all([0 <= x < 10 for x in seq])
single_digit = lambda seq: np.all([0 <= x < 10 for x in seq])

# Return true if sequence is not empty
non_empty = lambda seq: len(seq) > 0

# RNN filter
rnn_filter = lambda seq: len(seq) > 2 and np.all([0 <= x < 1000 for x in seq])
rnn_filterv2 = lambda seq: len(seq) > 2 and np.all([0 <= x < 2000 for x in seq])

# Cast to int
int_cast = lambda seq: [int(x) for x in seq]

# Not exactly what's needed
def compose(*filters):
    def inner(arg):
        for f in reversed(filters):
            arg = f(arg)
        return arg
    return inner