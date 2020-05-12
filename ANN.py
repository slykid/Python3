import numpy as np

class perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])

        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)

            self.errors.append(errors)

        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))  # 0.0


## ex. iris
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
iris.tail

y = iris.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

x = iris.iloc[0:100, [0, 2]].values

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label = 'setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label = 'versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors) + 1),
         ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.show()


# 아달린 알고리즘 API
import numpy as np

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.costs = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activate(net_input)
            errors = y - output
            self.w[1:] += self.eta * x.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)

        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def activate(self, x):
        return x

    def predict(self, x):
        return np.where(self.activate(self.net_input(x)) >= 0.0, 1, -1)


# 학습률 설정 비교
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
iris.tail

x = iris.iloc[0:100, [0, 2]].values

y = iris.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(x, y)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(x, y)

ax[0].plot(range(1, len(ada1.costs) + 1), np.log10(ada1.costs), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-Squared Error)')
ax[0].set_title('Adaline - Learning_rate = 0.01')

ax[1].plot(range(1, len(ada2.costs) + 1), np.log10(ada2.costs), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-Squared Error')
ax[1].set_title('Adaline - Learning_rate = 0.0001')

# 표준화
x_std = np.copy(x)
x_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

ada_standard = AdalineGD(n_iter=15, eta=0.01)
ada_standard.fit(x_std, y)

plt.scatter(x_std[:50, 0], x_std[:50, 1], color='red', marker='o', label = 'setosa')
plt.scatter(x_std[50:100, 0], x_std[50:100, 1], color='blue', marker='x', label = 'versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada_standard.costs) + 1), ada_standard.costs, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.show()


## ex. MNIST
import os
import struct
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def load_mnist(path, kind="train"):
    label_path = os.path.join(path, '%s-labels-idx1-ubyte' %kind)
    image_path = os.path.join(path, '%s-images-idx3-ubyte' %kind)

    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        # > : 빅 엔디언을 의미
        # I : 부호가 없는 정수를 의미

        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

x_train, y_train = load_mnist('dataset/MNIST/raw', kind='train')
x_test, y_test = load_mnist('dataset/MNIST/raw', kind='t10k')
print("행 : %d, 열 : %d\n" % (x_train.shape[0], x_train.shape[1]))
print("행 : %d, 열 : %d\n" % (x_test.shape[0], x_test.shape[1]))

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()

np.savez_compressed('dataset/MNIST/mnist_scaled.npz', X_train=x_train, Y_train=y_train,
                    X_test=x_test, Y_test=y_test)

import numpy as np
mnist = np.load('dataset/MNIST/mnist_scaled.npz')
print(mnist.files)

x_train = mnist['X_train']
y_train = mnist['Y_train']
x_test = mnist['X_test']
y_test = mnist['Y_test']

## MLP 학습하기
from MLP import NeuralNetMLP

nn = NeuralNetMLP(
                    n_hidden=100,
                    l2=0.01,
                    epochs=200,
                    eta=0.0005,
                    minibatch_size=100,
                    shuffle=True,
                    seed=1
                  )

nn._fit(
            x_train=x_train[:55000],
            y_train=y_train[:55000],
            x_valid=x_train[55000:],
            y_valid=y_train[55000:]
        )