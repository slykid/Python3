# 동적계획법으로 코딩
import time
import numpy as np

def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)

class Sigmoid:
    def __init__(self):
        self.last_o = 1

    def __call__(self, x):
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o

    def grad(self):
        # sigmoid(x) * (1 - sigmoid(x))
        return self.last_o * (1 - self.last_o)

class MeanSquaredError:
    def __init__(self):
        self.dh = 1  # gradient initial
        self.last_diff = 1


    def __call__(self, h, y):
        # 1/2 * mean((h - y) ^ 2)
        self.last_diff = h - y
        return np.mean(np.square(self.last_diff)) / 2

    def grad(self):
        # h - y
        return self.last_diff

class Neuron:
    def __init__(self, W, b, activate_obj):
        # Model params initialize
        self.W = W
        self.b = b
        self.activate_obj = activate_obj()

        # Gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))

        self.last_x = np.zeros((self.W.shape[0]))  # grad_W 를 계산하기 위한 변수

    def __call__(self, x):
        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b

        return self.activate_obj(self.last_h)

    def grad(self):
        # dy/dx = W
        return self.W * self.activate_obj.grad()

    def grad_W(self, dh):
        # dy/dW = x
        # dh = 이전까지의 그레디언트
        grad = np.ones_like(self.W)
        grad_a = self.activate_obj.grad()

        for j in range(grad.shape[1]):  # y = w^Tx + b
            grad[:, j] = dh[j] * grad_a[j] * self.last_x

        return grad

    def grad_b(self, dh):
        # dy/dh = 1
        # dh = 이전까지의 그레디언트
        return dh * self.activate_obj.grad() * 1

class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activate=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()

        # First Hidden Layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activate))

        # Hidden Layer
        for _ in range(hidden_depth - 1):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activate))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activate))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)

        return x

    def calc_gradient(self, loss_obj):
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)

        for i in range(len(self.sequence) - 1, 0, -1):
            l1 = self.sequence[i]
            l0 = self.sequence[i - 1]

            l0.dh = _m(l0.grad(), l1.dh)
            l0.dW = l0.grad_W(l1.dh)
            l0.db = l0.grad_b(l1.dh)

        self.sequence.remove(loss_obj)

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)
    network.calc_gradient(loss_obj)

    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db

    return loss

#######################################

x = np.random.normal(0.0, 1.0, (10, ))
y = np.random.normal(0.0, 1.0, (2, ))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activate=Sigmoid)
loss_obj = MeanSquaredError()

for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)

    print(f"Epoch {epoch+1}: Test loss {loss}")
print(f"{time.time() - t} seconds elapsed.")

# Epoch 1: Test loss 2.1665131983990324
# Epoch 2: Test loss 2.142156892336417
# ...
# Epoch 100: Test loss 1.3813622629762983
# 0.058008432388305664 seconds elapsed.