import numpy as np

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("MacOSX")

function_x = np.linspace(-3, 3, 100)
function_y = 2 * function_x ** 2

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(function_x, function_y)

x = 2
lr = 0.1  # learning-rate

# 원래함수는 y=2x^2 였다고 가정함
for _ in range(10):
    dy_dx = 4 * x
    x = x - lr * dy_dx
    y = 2 * x ** 2

    # 점차적으로  감소하는 현상을 확인 가능
    ax.scatter(x, y, color='red', s=100)

# 만약, learning rate 을 1로 준다면? (원래함수는 y=2x^2 였다고 가정함)
lr = 1  # learning-rate

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(function_x, function_y)

for _ in range(10):
    dy_dx = 4 * x
    x = x - lr * dy_dx
    y = 2 * x ** 2

    # 점차적으로  감소하는 현상을 확인 가능
    ax.scatter(x, y, color='red', s=100)

# Gradient 값이 항상 점진적으로 줄어들지만은 않는다.
# - y 값만 놓고 보면 점진적으로 줄어드는 것을 볼 수 있지만, 그에 대응하는 x 값은 지그재그 형식이 될 수 있음
function_x = np.linspace(-3, 3, 100)
function_y = 2 * function_x ** 2

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(function_x, function_y)

x = 3
y = 2 * x ** 2
lr = 0.4  # learning-rate

ax.scatter(x, y, color='red', s=300)

for _ in range(30):
    dy_dx = 4 * x
    x = x - lr * dy_dx
    y = 2 * x ** 2

    # 점차적으로  감소하는 현상을 확인 가능
    ax.scatter(x, y, color='red', s=100)


# 시작 지점이 음수여도 양수일 때와 유사함
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(function_x, function_y)

x = -2
y = 2 * x ** 2
lr = 0.1  # learning-rate

ax.scatter(x, y, color='red', s=300)

for _ in range(30):
    dy_dx = 4 * x
    x = x - lr * dy_dx
    y = 2 * x ** 2

    # 점차적으로  감소하는 현상을 확인 가능
    ax.scatter(x, y, color='red', s=100)


# 실제 문제에서는 아래와 같은 상황임
# - 미분된 값만을 이용해서 y의 최소값을 찾는 상황
x = 3
y = 2 * x ** 2
lr = 0.01  # learning-rate

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y, color='red', s=300)

for _ in range(50):
    dy_dx = 4 * x
    x = x - lr * dy_dx
    y = 2 * x ** 2

    # 점차적으로  감소하는 현상을 확인 가능
    ax.scatter(x, y, color='red', s=100)
