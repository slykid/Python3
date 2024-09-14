import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colormaps

np.random.seed(1)
matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

# Set Params
N = 100
lr = 0.01
t_w, t_b = 5, -3
w, b = np.random.uniform(-3, 3, 2)

# Generate Dataset
x_data = np.random.randn(N, )
y_data = x_data * t_w + t_b
# y_data += 0.1 * np.random.randn(N, )  # Noise 추가

# Visualize Dataset
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_data, y_data)

cmap = colormaps.get_cmap('rainbow', lut=N)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_data, y_data)

x_range = np.array([x_data.min(), x_data.max()])
J_track = []
w_track, b_track = [], []
for idx, (x, y) in enumerate(zip(x_data, y_data)):
    w_track.append(w)
    b_track.append(b)

    # Visualize updated model
    y_range = w * x_range + b
    ax.plot(x_range, y_range, color=cmap(idx), alpha=0.3)

    # Forward Propagation
    pred = x * w + b
    J = (y - pred) ** 2
    J_track.append(J)

    # Jacobians
    dJ_dpred = -2 * (y - pred)
    dpred_dw = x
    dpred_db = 1

    # Back Propagation
    dJ_dw = dJ_dpred * dpred_dw
    dJ_db = dJ_dpred * dpred_db

    # Parameter Update
    w = w - lr * dJ_dw  # = w + 2 * lr * x * (y - pred) 를 넣은 것과 동일한 결과가 나옴
    b = b - lr * dJ_db  # = b + 2 * lr * (y - pred) 를 넣은 것과 동일한 결과가 나옴

# Visualize result
fig, ax = plt.subplots(2, 1, figsize=(20, 10))
ax[0].plot(J_track)
ax[1].plot(w_track, color='darkred')
ax[1].plot(b_track, color='darkblue')

ax[0].set_ylabel('MSE', fontsize=30)
ax[0].tick_params(labelsize=20)

ax[1].axhline(y=t_w, color='darkred', linestyle=':')
ax[1].axhline(y=t_b, color='darkblue', linestyle=':')
ax[1].tick_param(labelsize=20)
