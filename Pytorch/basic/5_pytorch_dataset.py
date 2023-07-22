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

from copy import deepcopy

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Data load
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

df.tail()
df.columns

data = torch.from_numpy(df.values).float()
data.shape  # torch.Size([569, 31])

# Preprocessing
x = data[:, :10]
y = data[:, -1:]
print("x.shape: " + str(x.shape) + ", y.shape: " + str(y.shape))

ratio = [.6, .2, .2]  # train, valid, test 순

train_cnt = int(data.size(0) * ratio[0])
valid_cnt = int(data.size(0) * ratio[1])
test_cnt = int(data.size(0)) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]
print("Train: %d / Valid: %d / Test: %d" % (train_cnt, valid_cnt, test_cnt))

indices = torch.randperm(data.size(0))

x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

x = torch.split(x, cnts, dim=0)
y = torch.split(y, cnts, dim=0)

for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())

# Set Hyperparmeter
n_epochs = 10000
batch_size = 128
print_interval = 500
early_stop = 100

# Make dataloader
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

# Modeling
model = nn.Sequential(
    nn.Linear(x[0].size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y[0].size(-1)),
    nn.Sigmoid(),
)

model

optimizer = optim.Adam(model.parameters())


# Train model
lowest_loss = np.inf
lowest_epoch = np.inf
best_model = None

train_history, valid_history = [], []

for i in range(n_epochs):
    model.train()

    train_loss, valid_loss = 0, 0
    y_hat = []

    for x_i, y_i in train_loader:
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss += float(loss)

    train_loss = train_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        valid_loss = 0

        for x_i, y_i in valid_loader:
            y_hat_i = model(x_i)
            loss = F.binary_cross_entropy(y_hat_i, y_i)

            valid_loss += float(loss)

            y_hat += [y_hat_i]

    valid_loss = valid_loss / len(valid_loader)

    train_history += [train_loss]
    valid_history += [valid_loss]

    if (i + 1) % print_interval == 0:
        print("Epoch %d: train loss = %.4e    valid_loss = %.4e    lowest_loss = %.4e" % (i + 1, train_loss, valid_loss, lowest_loss))

    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i

        best_model = deepcopy(model.state_dict())

    else:
        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
            print("There is no improvement during last %d epochs." % early_stop)
            break

print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))
print("The Best model: \n")
model.load_state_dict(best_model)

# Check Loss History
plot_from = 2
# plt.figure(figsize=(20, 10))
plt.grid(True)
plt.title("Train / Valid Loss History")
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:],
)
plt.yscale('log')
plt.show()


# Make prediction
test_loss = 0
y_hat = []

model.eval()
with torch.no_grad():
    for x_i, y_i in test_loader:
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i)

        test_loss += loss

        y_hat += [y_hat_i]

test_loss = test_loss / len(test_loader)
y_hat = torch.cat(y_hat, dim=0)

print("Test loss: %.4e" % valid_loss)

correct_cnt = (y[2] == (y_hat > .5)).sum()
total_cnt = float(y[2].size(0))

print("Test Accuracy: %.4f" % (correct_cnt / total_cnt))
