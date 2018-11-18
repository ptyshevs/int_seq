import pandas as pd
from tools import seq_to_num, acc_score
import sys
sys.path.append("..")
from models import mark_chain

k_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
minlen = [6, 12, 15, 18, 20, 25, 30, 45, 55, 65]
unique = [2, 3, 4, 5, 6, 7, 8, 9, 10]
kk = list()
uni = list()
minll = list()

df_train = pd.read_csv('../data/kaggle_train.csv', index_col=0)
train_X, train_y = seq_to_num(df_train.Sequence, pad=False)
scores = list()
counts = list()
i = 1
for k in k_:
    for minl in minlen:
        for un in unique:
            if markov_model.bad_model:
                continue
            markov_model = mark_chain.MarkovChains(max_unique_count=un, minlen=minl, slice=-1, k=k)
            kk.append(k)
            uni.append(un)
            minll.append(minl)
            ind, sequences, pred = markov_model.predict(train_X)
            score = acc_score(pred, train_y[ind])
            scores.append(score)
            counts.append(len(pred))
            print(f"Iteration {i}, score: {score}")
            i += 1
d = {'Accuracy': scores, 'Count': counts, 'Max_unique_el': uni, 'Min_lenght_seq': minll, 'k': kk}
data = pd.DataFrame(data=d)
data.to_csv('Hyper_parameters_Markov_Chain.csv')
