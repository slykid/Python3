import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import requests
import ssl

# 사용 변수 선언
CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
EPOCHS = 50
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = optimizers.RMSprop()

# 사용함수 정의
# Data loader
def load_cifar_10():
    # Trouble Shooting
    # 에러: ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1108)
    requests.packages.urllib3.disable_warnings()

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    # Data Load
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # 전처리
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # 정규화
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std +1e-7)

    y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, CLASSES)

    return (x_train, y_train), (x_test, y_test)

# Custom CNN 모델 정의1. 얕은 신경망
def CustomSwallowCNN(input_shape, classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation="softmax"))

    return model

def CustomDeepCNN(input_shape, classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:], activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(CLASSES, activation="softmax"))

    return model

(x_train, y_train), (x_test, y_test) = load_cifar_10()

# Swallow CNN
model = CustomSwallowCNN(input_shape=INPUT_SHAPE, classes=CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs/20211004/CustomSwallowCNN")]

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT,callbacks=callbacks)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print("\nTest Score:", score[0])  # 1.4943
print("\nTest Accuracy:", score[1])  # 0.668

# Deep CNN
model2 = CustomDeepCNN(input_shape=INPUT_SHAPE, classes=CLASSES)
model2.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs/20211004/CustomDeepCNN")]

model2.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT,callbacks=callbacks)

score2 = model2.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print("\nTest Score:", score2[0])  # 0.5689
print("\nTest Accuracy:", score2[1])  # 0.8454