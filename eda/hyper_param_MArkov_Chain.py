import pandas as pd
from eda import tools
from models import mark_chain

k_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
minlen = [6, 12, 15, 18, 20, 25, 30, 45, 55, 65]
unique = [2, 3, 4, 5, 6, 7, 8, 9, 10]
kk = list()
uni = list()
minll = list()

df_train = pd.read_csv('train.csv', index_col=0)
train_X, train_y = tools.seq_to_num(df_train.Sequence, pad=False)
scores = list()
counts = list()
i = 1
for k in k_:
    for minl in minlen:
        for un in unique:
            kk.append(k)
            uni.append(un)
            minll.append(minl)
            markov_model = mark_chain.MarkovChains(max_unique_count=un, minlen=minl, slice=-1, k=k)
            ind, sequences, pred = markov_model.predict(train_X)
            score = tools.acc_score(pred, train_y[ind])
            scores.append(score)
            counts.append(len(pred))
            print(i)
            i += 1
d = {'Accuracy': scores, 'Count': counts, 'Max_unique_el': uni, 'Min_lenght_seq': minll, 'k': kk}
data = pd.DataFrame(data=d)
data.to_csv('Hyper_parameters_Markov_Chain.csv')
