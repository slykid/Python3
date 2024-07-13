import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model

# 1.1 IO of Softmax
logits = tf.random.uniform(shape=(2, 5), minval=-10, maxval=10)

softmax_value = Activation('softmax')(logits)
softmax_sum = tf.reduce_sum(softmax_value, axis=1) # 각 열 단위로 합을 계산!

print("Logits: ", logits.numpy())
print("Probabilities: ", softmax_value.numpy())
print("Sum of softmax value: ", softmax_sum)

# 1.2 Softmax in Dense Layers
logits = tf.random.uniform(shape=(8, 5), minval=-10, maxval=10)

dense = Dense(units=8, activation='sigmoid')

Y = dense(logits)
print(tf.reduce_sum(Y, axis=1))

# 2.1 Multi-class Classifier
class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()

        self.dense1 = Dense(units=8, activation='relu')
        self.dense2 = Dense(units=5, activation='relu')
        self.dense3 = Dense(units=3, activation='sigmoid')

    def call(self, x):
        print("X: {}\n".format(x.numpy()))

        x = self.dense1(x)
        print("A1: {}\n".format(x.numpy()))

        x = self.dense2(x)
        print("A2: {}\n".format(x.numpy()))

        x = self.dense3(x)
        print("Y: {}\n".format(x.numpy()))
        print("Sum of vectors: {}\n".format(tf.reduce_sum(x, axis=1)))

        return x

model = TestModel()

logits = tf.random.uniform(shape=(8, 5), minval=-10, maxval=10)
Y = model.call(logits)