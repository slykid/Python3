import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

df.tail()
df.columns

data = torch.from_numpy(df.values).float()
data.shape  # torch.Size([569, 31])

x = data[:, :10]
y = data[:, :-1]
print("x.shape: " + x.shape ", y.shape: " + y.shape)

ratio = [.6, .2, .2]  # train, valid, test 순

train_cnt = int(data.size(0) * ratio[0])
valid_cnt = int(data.size(0) * ratio[1])
test_cnt = int(data.size(0) * ratio[2])
cnts = [train_cnt, valid_cnt, test_cnt]
print("Train: %d / Valid: %d / Test: %d" % (train_cnt, valid_cnt, test_cnt))

indices = torch.randperm(data.size(0))

x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

x = x.split(cnts, dim=0)
y = y.split(cnts, dim=0)

for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())

