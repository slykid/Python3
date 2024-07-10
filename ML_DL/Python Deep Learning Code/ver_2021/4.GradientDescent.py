import numpy as np
import tensorflow as tf

# 하이퍼파라미터 설정
EPOCHS = 200

# 네트워크 정의
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.d1(x)
        return self.d2(x)

# 학습 루프 정의
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):

    # 그레디언트 계산
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)  # df(x) / dx
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_metric(labels, predictions)

# 데이터 셋 생성 & 전처리
np.random.seed(0)

## 각 10개의 무작위 점들에 대해서 구분하는 모델 생성 예정
pts = list()
labels = list()
center_pts = np.random.uniform(-8.0, 8.0, (10, 2))

for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.rand(*center_pt.shape))
        labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32)  # GPU를 사용할 경우 float32 형으로 변환해서 넣어줘야함
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# 학습 루프
for epoch in range(EPOCHS):
    for x, y in train_ds:
        train_step(model, x, y, loss_object, optimizer, train_loss, train_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100))

# 데이터셋 및 학습 파라미터 저장
np.savez_compressed('ch04_dataset.npz', inputs=pts, labels=labels)

w_h, b_h = model.d1.get_weights()
w_o, b_o = model.d2.get_weights()
w_h = np.transpose(w_h)
w_o = np.transpose(w_o)
np.savez_compressed('ch04_parameters.npz', W_h = w_h, b_h=b_h, W_o=w_o, b_o=b_o)