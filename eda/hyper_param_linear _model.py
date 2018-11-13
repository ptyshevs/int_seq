import pandas as pd
import time
import try_linear_model
import itertools

t = time.time()
train = pd.read_csv('train.csv')
y = try_linear_model.get_y(train)
# choose hyper parametres
# max_length = list([6, 10, 12, 14, 15, 20])
# slices = list([12, 12, 15, 17, 20, 22])
# min_lenght = list([8, 10, 12, 14, 16, 18])
max_length = list()
slices = list()
min_lenght = list()
stuff = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
for subset in itertools.combinations(stuff, 3):
    max_length.append(subset[0])
    slices.append(subset[1])
    min_lenght.append(subset[2])

scores = list()
counts = list()
for max, sl, min in zip(max_length, slices, min_lenght):
    pred_values = try_linear_model.make_prediction(train, y, max_length=max, slice=sl, min_length=min)
    score, count = try_linear_model.score(y, pred_values)
    scores.append(round(score, 3))
    counts.append(count)
d = {'Accuracy': scores, 'Count': counts, 'Slices': slices, 'Min_lenght_seq': min_lenght, 'Max_lenght_elem': max_length}
data = pd.DataFrame(data=d)
data.to_csv('Hyper_parametres_10_20_linear_model.csv')
print(time.time() - t)
