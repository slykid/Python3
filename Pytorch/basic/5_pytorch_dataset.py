import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

df.tail()
df.columns

data = torch.from_numpy(df.values).float()
data.shape  # torch.Size([569, 31])

x = data[:, :10]
y = data[:, -1:]
print("x.shape: " + str(x.shape) + ", y.shape: " + str(y.shape))

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

# Set Hyperparmeter
n_epochs = 10000
batch_size = 128
print_interval = 500
early_stop = 100

train_loader = DataLoader(
    dataset=CustomDataset(x[0], y[0]),
    batch_size=batch_size,
    shuffle=True
)

valid_loader = DataLoader(
    dataset=CustomDataset(x[1], y[1]),
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    dataset=CustomDataset(x[2], y[2]),
    batch_size=batch_size,
    shuffle=False
)

print("Train %d / Valid %d / Test %d" % (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))


