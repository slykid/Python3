import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# 학습 변수 선언
EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = optimizers.Adam()
VALIDATION_SPLIT = 0.95

IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
CLASSES = 10

# LeNet 신경망
def LeNet(input_shape, classes):
    model = models.Sequential()

    # CONV1 : ReLU -> Max Pooling / 5 x 5 x 20 / ReLU / input_shape
    model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # CONV2 : ReLU -> Max Pooling / 5 x 5 x 50 / ReLU / input_shape
    model.add(layers.Conv2D(50, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # FC(Fully-Connected) Layer
    model.add(layers.Flatten())  # 컨볼루션 결과를 1차 행렬로 펴주는 작업
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    return model

# 학습하기
## 데이터 로드
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

## 전처리
### 1) 크기조정
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

### 2) 정규화
x_train, x_test = x_train/255.0, x_test/255.0

### 3) 형식 변환
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

### 4) 클래스를 이진 벡터로 변환
y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, CLASSES)

## 모델 초기화
model = LeNet(input_shape=INPUT_SHAPE, classes=CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
model.summary()

## 텐서보드 사용
callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs")]

history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks
)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("\nTest Score: ", score[0])
print("\nTest Accuracy: ", score[1])
