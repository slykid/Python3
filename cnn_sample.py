import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

# load datasets
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

# check data shape
train_x.shape, train_y.shape  # ((60000, 28, 28), (60000,))
test_x.shape, test_y.shape  # ((10000, 28, 28), (10000,))

# increasse dimension
# gray scale 인 경우에 차원수를 증가시켜줘야함
# train_x = tf.expand_dims(train_x, -1)  # 기존에 사용하던 방법
# test_x = tf.expand_dims(test_x, -1)  # 기존에 사용하던 방법
train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

train_x.shape, train_y.shape  # ((60000, 28, 28, 1), (60000,))
test_x.shape, test_y.shape  # ((10000, 28, 28, 1), (10000,))

# one-hot encoding
# train_y_new = tf.one_hot(train_y, 10)
# test_y_new = tf.one_hot(test_y, 10)
# train_y_new.shape
# test_y_new.shape

# rescaling
np.min(train_x), np.max(train_x)

train_x  = train_x / 255.
test_x  = test_x / 255.

# build model
inputs = layers.Input((28, 28, 1))
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)  # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

model.summary()  # 배치 사이즈를 지정하지 않았기 때문에 None 으로 표시됨

# loss function
# loss = 'binary_crossentropy'
# loss = 'categorical_crossentropy'

# tf.keras.losses.sparse_categorical_crossentropy  # one hot encoding 이 안된 경우 ex) [1, 3, 5] 와 같이 사용가능
# tf.keras.losses.categorical_crossentropy  # one hot encoding 이 된 경우
# tf.keras.losses.binary_crossentropy  # 클래스가 이진인 경우

loss_func = tf.losses.sparse_categorical_crossentropy

# Metrics
metrics = [tf.keras.metrics.sparse_categorical_accuracy]  # recall, precision 등 여러개를 주기 위해 list 형식으로 선언

# Optimizer
opt = tf.keras.optimizers.Adam()

# Compile
model.compile(optimizer=opt,
              loss=loss_func,
              metrics=metrics)

# training
# set hyperparameter
num_epochs = 1  # 학습 횟수
batch_size = 32  # 메모리를 효율적으로 사용하기 위해서 설정

# fit model
hist = model.fit(train_x, train_y,
                  batch_size=batch_size,
                  shuffle=True,
                  epochs=num_epochs)

print(hist)