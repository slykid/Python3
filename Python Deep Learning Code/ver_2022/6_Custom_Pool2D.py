# 1. Max/Average Pooling
# 1) Max Pooling
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPooling1D

L, f, s = 10, 2, 1

x = tf.random.normal(shape=(1, L, 1))
pool_max = MaxPooling1D(pool_size = f, strides=s)
pooled_max = pool_max(x)

print("x: {}\n{}".format(x.shape, x.numpy().flatten()))
print("pooled_max(Tensorflow): {}\n".format(pooled_max.shape, pooled_max.numpy().flatten()))

x = x.numpy().flatten()
pool_max_man = np.zeros((L - f + 1, ))
for i in range(L - f + 1):
    window = x[i : i + f]
    pool_max_man[i] = np.max(window)

print("Pooled Max(manual) : {}\n{}".format(pool_max_man.shape, pool_max_man))
# Pooled Max(manual) : (9,)
# [ 1.22528744 -0.11209985 -0.11209985 -1.51651692  0.49176407  2.62664771
#   2.62664771 -0.05767371  0.50635576]

# 2) Average Pooling
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import AveragePooling1D

L, f, s = 10, 2, 1

x = tf.random.normal(shape=(1, L, 1))
pool_mean = AveragePooling1D(pool_size = f, strides=s)
pooled_mean = pool_mean(x)

print("x: {}\n{}".format(x.shape, x.numpy().flatten()))
print("pooled_avg(Tensorflow): {}\n{}".format(pooled_mean.shape, pooled_mean.numpy().flatten()))

x = x.numpy().flatten()
pool_avg_man = np.zeros((L - f + 1, ))
for i in range(L - f + 1):
    window = x[i : i + f]
    pool_avg_man[i] = np.mean(window)

print("Pooled Max(manual) : {}\n{}".format(pool_avg_man.shape, pool_avg_man))

# 2. 2D Max/Avg Pooling
# 1) 2D Max Pooling
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D

N, n_H, n_W, n_C = 1, 5, 5, 1
f, s = 2, 1

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
pool_max = MaxPooling2D(pool_size=f, strides=s)
pooled_max = pool_max(x)

print("x: {}\n{}".format(x.shape, x.numpy().squeeze()))
print("pooled_max(Tensorflow): {}\n{}".format(pooled_max.shape, pooled_max.numpy().squeeze()))

# Manual
x = x.numpy().squeeze()
pooled_max_man = np.zeros(shape=(n_H - f + 1, n_W - f + 1))

for i in range(n_H - f + 1):
    for j in range(n_W - f + 1):
        window = x[i:i + f, j:j + f]
        pooled_max_man[i, j] = np.max(window)

print("pooled_max(Manual): {}\n{}".format(pooled_max_man.shape, pooled_max_man))

# 2) 2D Avg Pooling
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import AveragePooling2D

N, n_H, n_W, n_C = 1, 5, 5, 1
f, s = 2, 1

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
pool_avg = AveragePooling2D(pool_size=f, strides=s)
pooled_avg = pool_avg(x)

print("x: {}\n{}".format(x.shape, x.numpy().squeeze()))
print("pooled_avg(Tensorflow): {}\n{}".format(pooled_avg.shape, pooled_avg.numpy().squeeze()))

# Manual
x = x.numpy().squeeze()
pooled_avg_man = np.zeros(shape=(n_H - f + 1, n_W - f + 1))

for i in range(n_H - f + 1):
    for j in range(n_W - f + 1):
        window = x[i:i + f, j:j + f]
        pooled_avg_man[i, j] = np.mean(window)

print("pooled_avg(Manual): {}\n{}".format(pooled_avg_man.shape, pooled_avg_man))

# 3. 3D Max/Avg Pooling
# 1) 3D Max Pooling
import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D

N, n_H, n_W, n_C = 1, 5, 5, 3
f, s = 2, 2

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
print("x: {}\n{}".format(x.shape, np.transpose(x.numpy().squeeze(), (2, 0, 1))))

pool_max = MaxPooling2D(pool_size=f, strides=s)
pooled_max = pool_max(x)
pooled_max_t = np.transpose(pooled_max.numpy().squeeze(), (2, 0, 1))
print("pooled_max(Tensorflow): {}\n{}".format(pooled_max.shape, pooled_max_t))

# Manual
x_sqz = x.numpy().squeeze()
n_h = math.floor((n_H - f) / s + 1)
n_w = math.floor((n_W - f) / s + 1)
print(n_h, n_w)

pooled_max_man = np.zeros(shape=(n_h, n_w, n_C))

for c in range(n_C):
    c_image = x_sqz[:, :, c]
    _h = 0
    for h in range(0, n_H - f + 1, s):
        _w = 0

        for w in range(0, n_W - f + 1, s):
            window = c_image[h:h + f, w:w + f]
            pooled_max_man[_h, _w, c] = np.max(window)

            _w += 1

        _h += 1

pooled_max_man_t = np.transpose(pooled_max_man, (2, 0, 1))
print("pooled_max(Manual): {}\n{}".format(pooled_max_man.shape, pooled_max_man_t))

# 4. Padding
# 1) ZeroPadding2D
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import ZeroPadding2D

images = tf.random.normal(shape=(1, 5, 5, 3))
print(np.transpose(images.numpy().squeeze(), (2, 0, 1)))
print(images.shape)

zero_padding = ZeroPadding2D(padding=2)
y = zero_padding(images)

print(np.transpose(y.numpy().squeeze(), (2, 0, 1)))
print(y.shape)

# 2) ZeroPadding with Conv2D Layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

images = tf.random.normal(shape=(1, 5, 5, 3))
conv = Conv2D(filters=1, kernel_size=3, padding='valid')
y = conv(images)
print(y.shape)

# 5. Strides
# 1) Strides in Conv2D Layers
import tensorflow as tf

from tensorflow.keras.layers import Conv2D

images = tf.random.normal(shape=(1, 5, 5, 3))
conv = Conv2D(filters=1, kernel_size=3, padding='valid', strides=2)
y = conv(images)
print(images.shape)
print(y.shape)

# 2) Strides in Pooling Layers
import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D

images = tf.random.normal(shape=(1, 5, 5, 3))
conv = MaxPooling2D(pool_size=3, strides=2)
y = conv(images)
print(images.shape)
print(y.shape)
