import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

# 1.1 Graphs of Odds and Logit
p_np = np.linspace(0.01, 0.99, 100)
p_tf = tf.linspace(0.01, 0.99, 100)

print(p_np)
print(p_tf)

odds_np = p_np/(1-p_np)
odds_tf = p_tf/(1-p_tf)

logit_np = np.log(odds_np)
logit_tf = tf.math.log(odds_tf)

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
axes[0].plot(p_np, odds_np)
# axes[0].plot(p_tf, odds_tf)
axes[1].plot(p_np, logit_np)
# axes[1].plot(p_tf, logit_tf)

xticks = np.arange(0, 1.1, 0.2)
axes[0].tick_params(labelsize=15)
axes[0].set_xticks(xticks)
axes[0].set_ylabel('Odds', fontsize=20, color='darkblue')
axes[1].tick_params(labelsize=15)
axes[1].set_xticks(xticks)
axes[1].set_ylabel('Logits', fontsize=20, color='darkblue')
axes[1].set_xlabel('Probability', fontsize=20, color='darkblue')

# 1.2 Graphs of Sigmoid
X = tf.linspace(-10, 10, 100)
sigmoid = Activation('sigmoid')(X)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X.numpy(), sigmoid.numpy())


# 2.1 Single Variate Logistic Regression
X = tf.random.normal(shape=(100, 1))
dense = Dense(units=1, activation='sigmoid')

Y = dense(X)
print(Y.shape)

fig, ax=plt.subplots(figsize=(7, 7))
ax.scatter(X.numpy().flatten(), Y.numpy().flatten())

# 2.2 Multi Variate Logistic Regression
X = tf.random.normal(shape=(100, 5))
dense = Dense(units=1, activation='sigmoid')

Y = dense(X)
print(Y.shape)

# 2.3 Binary Classification
model = Sequential()
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
