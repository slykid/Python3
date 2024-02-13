# 직접 작성한 모듈이 import 되지 않을 경우
# 1. File - Settings - Project Structure 에서 Source 로 폴더 추가하기
# 2. 폴더 우클릭 - Mark Directory as 에서 Source로 폴더 추가하기

# 라이브러리 간의 충돌이 발생하는 것을 방지하기 위함
# 딥러닝의 경우 발생할 수 있는 이슈이므로 추가하는 습관을 들일 것
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim

from utils import load_mnist
from trainer import Trainer
from model import Autoencoder

from argparse import Namespace

config = {
    'train_ratio': .8,
    'batch_size': 256,
    'n_epochs': 50,
    'verbose': 1,
    'btl_size': 2
}

config = Namespace(**config)
print(config)

def show_image(x):
    if x.dim() == 1:
        x = x.view(int(x.size(0) ** .5), -1)
    plt.imshow(x, cmap='gray')
    plt.show()

x_train, y_train = load_mnist(flatten=True)
x_test, y_test = load_mnist(is_train=False, flatten=True)

train_cnt = int(x_train.size(0) * config.train_ratio)
valid_cnt = x_train.size(0) - train_cnt

# Shuffle dataset (Train - Valid)
index = torch.randperm(x_train.size(0))
x_train, x_valid = torch.index_select(x_train, dim=0, index=index).split([train_cnt, valid_cnt], dim=0)
y_train, y_valid = torch.index_select(y_train, dim=0, index=index).split([train_cnt, valid_cnt], dim=0)

print("Train: ", x_train.shape, y_train.shape)
print("Valid: ", x_valid.shape, y_valid.shape)
print("Test: ", x_test.shape, y_test.shape)

# Model Object & Optimizer, Criterion Settings
model = Autoencoder(btl_size=config.btl_size)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Model Train
trainer = Trainer(model, optimizer, criterion)
trainer.train((x_train, x_train), (x_valid, x_valid), config)  # Encoder - Decoder 구조이기 때문에 x 에 대한 것만 사용

# Model Test
with torch.no_grad():
    import random

    idx = int(random.random() * x_test.size(0))

    recon = model(x_test[idx].view(1, -1)).squeeze()

    show_image(x_test[idx])
    show_image(recon)

# Hidden Space
if config.btl_size == 2:
    color_map = ['brown', 'red', 'orange', 'yellow', 'green',
                 'blue', 'navy', 'purple', 'gray','black', ]

    plt.figure(figsize=(20, 10))
    with torch.no_grad():
        latents = model.encoder(x_test[:1000])

        for i in range(10):
            target_latents = latents[y_test[:1000] == i]
            y_target = y_test[:1000][y_test[:1000] == i]

            plt.scatter(target_latents[:, 0],
                        target_latents[:, 1],
                        marker='o',
                        color=color_map[i],
                        label=i)

            plt.legend()
            plt.grid(axis='both')
            plt.show()

if config.btl_size == 2:
    min_range, max_range = -2., 2.
    n = 20
    step = (max_range - min_range) / float(n)

    with torch.no_grad():
        lines = []

        for v1 in np.arange(min_range, max_range, step):

            # |z| = (20, 2)
            z = torch.stack([
                torch.FloatTensor([v1] * n),
                torch.FloatTensor([v2 for v2 in np.arange(min_range, max_range, step)]),
            ], dim=-1)

            line = torch.clamp(model.decoder(z).view(n, 28, 28), 0, 1)  # decoder(|Z|) = (20, 784) -> (20, 28, 28)
            line = torch.cat([line[i] for i in range(n - 1, 0, -1)], dim=0)  # Hidden Space 에 표시된 내용과 같도록 할건데
                                                                             # 이미지의 경우 좌상단이 가장 작은 값으로 표시됨
                                                                             # -> 역순으로 for 문을 수행하는 이유임
            lines += [line]

        lines = torch.cat(lines, dim=-1)
        plt.figure(figsize=(20, 20))
        show_image(lines)