import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model

# 1. Model Implementation with Sequential Method
model = Sequential()
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=20, activation='sigmoid'))


# 2. Model subclassing
class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()

        self.dense1 = Dense(units=10, activation='sigmoid')
        self.dense2 = Dense(units=20, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        return x

model = TestModel()
print(model.dense1)
print(model.dense2)


# 3. Forward Propagation of Model
X = tf.random.normal(shape=(1, 10))

model1 = Sequential()
model1.add(Dense(units=10, activation='sigmoid'))
model1.add(Dense(units=20, activation='sigmoid'))

model2 = TestModel()

y1 = model1(X)  # Sequential Model
print(y1.numpy())

y2 = model2.call(X)
print(y2.numpy())

# 4. Layers in Models
print(type(model1.layers))

# 4.1 Layer 에서 제공해주는 변수 및 메소드 확인
layer = model1.layers[0]
for item in dir(layer):
    print(item)

# 4.2 Trainable Variables in Models
print(type(model1.trainable_variables))
print(len(model1.trainable_variables))
for variable in model1.trainable_variables:
    print(variable)  # Weight, Bias 를 의미
