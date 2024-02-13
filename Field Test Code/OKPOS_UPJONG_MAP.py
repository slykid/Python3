import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf  # tensorflow-gpu 1.12.3
import keras

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, Embedding
from keras.initializers import Constant
from keras.backend import tensorflow_backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# 사용 변수 정의
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
callbacks = [
    EarlyStopping(monitor='val_f1', patience=10, )
]

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    # value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)

        return value


def get_topN(pred, df):
    N = 3
    topN = np.argsort(pred, axis=-1)
    labels = np.array([[0, 0, 0]])
    probs = np.array([[0, 0, 0]])

    for idx, i in enumerate(pred):
        label = topN[idx][::-1][:N]
        prob = i[label][np.newaxis, :]
        probs = np.vstack((probs, prob))
        labels = np.vstack((labels, label))

    result = pd.concat([df.iloc[:, 0].reset_index(drop=True),pd.DataFrame(labels[1:]), pd.DataFrame(probs[1:])],axis=1, ignore_index=True).\
        rename(columns={0: "no", 1: "p1", 2: "p2", 3: "p3", 4: "confi_1", 5: "confi_2", 6: "confi_3"})
    print(result.head(1))

    return result

def custom_input(MAX_SEQUENCE_LENGTH, intput_layer, embedding_layer):
    embedded = embedding_layer(intput_layer)
    batchNorm = keras.layers.BatchNormalization()(embedded)

    return batchNorm

def custom_CNN(layer):
    cnn_1 = keras.layers.Conv1D(filters, kernel_size, activation='relu')(layer)
    cnn_2 = keras.layers.Conv1D(filters, kernel_size, activation='relu')(cnn_1)
    cnn_pool = MaxPooling1D(pool_size)(cnn_2)  # 여기서 CNN-LSTM

    return cnn_pool

def custom_LSTM(layer):
    # CUDA 10.2 / CuDNN 7.1.4 -> tensorflow-gpu 1.12.3 이 최적의 조건임
    lstm_1 = keras.layers.CuDNNLSTM(lstm_output_size, return_sequences=True)(layer)
    lstm_2 = keras.layers.CuDNNLSTM(lstm_output_size)(lstm_1)

    return lstm_2

def custom_CNN_LSTM(layer):
    cl_batch = keras.layers.BatchNormalization()(layer)

    # CUDA 10.2 / CuDNN 7.1.4 -> tensorflow-gpu 1.12.3 이 최적의 조건임
    cl_l1 = keras.layers.CuDNNLSTM(lstm_output_size, return_sequences=True)(cl_batch)
    cl_l2 = keras.layers.CuDNNLSTM(lstm_output_size)(cl_l1)

    return cl_l2

def subnet(input_layer, embedding_layer):
    # feature
    feature = custom_input(MAX_SEQUENCE_LENGTH, input_layer, embedding_layer)

    # CNN
    cnn_1 = custom_CNN(feature)
    cnn_flat = keras.layers.Flatten()(cnn_1)

    # LSTM
    lstm = custom_LSTM(feature)

    # CNN-LSTM
    cnn_lstm = custom_CNN_LSTM(cnn_1)

    # concat
    concat = keras.layers.Concatenate(axis=-1)([cnn_flat, lstm, cnn_lstm])

    return concat

def build_model():
    K.tf.reset_default_graph()

    # shared layer
    embedding_layer = keras.layers.Embedding(
        MAX_NUM_WORDS, EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True
    )

    # shop
    shop_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='shop')
    shop_net = subnet(shop_input, embedding_layer)

    # prod
    prod_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='prod')
    prod_net = subnet(prod_input, embedding_layer)
    concat = keras.layers.Concatenate(axis=-1)([shop_net, prod_net])
    preds = keras.layers.Dense(NUM_CLASSES, activation='softmax')(concat)
    model = keras.Model(inputs=[shop_input, prod_input], outputs=[preds])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1, 'acc']
    )

    return model

def model_one_feature():
    K.tf.reset_default_graph()

    # shared layer
    embedding_layer = keras.layers.Embedding(
        MAX_NUM_WORDS, EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True
    )

    # one feature
    one_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='shop')
    one_net = subnet(one_input, embedding_layer)
    preds = keras.layers.Dense(NUM_CLASSES, activation='softmax')(one_net)
    model = keras.Model(inputs=one_input, outputs=[preds])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1, 'acc'])

    return model

# 시작
MAX_SEQUENCE_LENGTH = 8
NUM_CLASSES = 327  # 변경되는 경우 있음
MAX_NUM_WORDS = 20000  # 조절
EMBEDDING_DIM = 100
kernel_size = 2
filters = 64
pool_size = 2
lstm_output_size = 200

print('MAX_NUM_WORDS: ', MAX_NUM_WORDS)

# Data load
print("Data load")

## train dataset
data = pd.read_csv("/data_ssd/menu_norm/data/train/data.csv", encoding="utf-8")

## test dataset
test_pdf = pd.read_csv("/data_ssd/menu_norm/data/test/data.csv", encoding="utf-8")

print("Tokenizing")

tk = Tokenizer(num_words=MAX_NUM_WORDS)
tk.fit_on_texts(data.sWord + data.pWord)
print(len(tk.word_index))

# features
feature_shop = pad_sequences(tk.texts_to_sequences(data.sWord.values), maxlen=MAX_SEQUENCE_LENGTH)
feature_prod = pad_sequences(tk.texts_to_sequences(data.pWord.values), maxlen=MAX_SEQUENCE_LENGTH)

print(min(data.label))
print(max(data.label))

label = to_categorical(np.asarray(data.label), num_classes=NUM_CLASSES)
print('Shape of f1 tensor:', feature_shop.shape)
print('Shape of f2 tensor:', feature_prod.shape)
print('Shape of label tensor:', label.shape)

# 검증셋 비율
VALIDATION_SPLIT = 0.1
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

feature_shop = feature_shop[indices]
feature_prod = feature_prod[indices]

label = label[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

shop_train = feature_shop[:-num_validation_samples]
prod_train = feature_prod[:-num_validation_samples]

y_train = label[:-num_validation_samples]

shop_val = feature_shop[-num_validation_samples:]
prod_val = feature_prod[-num_validation_samples:]

y_val = label[-num_validation_samples:]
print(len(test_pdf))

t_shop = pad_sequences(tk.texts_to_sequences(test_pdf.sWord.values), maxlen=MAX_SEQUENCE_LENGTH)
t_prod = pad_sequences(tk.texts_to_sequences(test_pdf.pWord.values), maxlen=MAX_SEQUENCE_LENGTH)
print(len(t_shop), len(t_prod))

# 모델링
print("1. 상호명\n")
K.clear_session()
K.tf.reset_default_graph()
model_shop = model_one_feature()
hist = model_shop.fit(
    shop_train, y_train,
    batch_size=4000, epochs=2000,
    validation_data=(shop_val, y_val),
    verbose=2,
    callbacks=callbacks
)

pred_shop = model_shop.predict(t_shop)
re_shop = get_topN(pred_shop, test_pdf)
re_shop.to_csv("/data_ssd/menu_norm/output/output_shop.csv", index=False)

print("2. 상품명\n")
K.clear_session()
K.tf.reset_default_graph()
model_prod = model_one_feature()
hist = model_prod.fit(
    prod_train, y_train,
    batch_size=4000, epochs=2000,
    validation_data=(prod_val, y_val),
    verbose=2,
    callbacks=callbacks
)

pred_prod = model_prod.predict(t_prod)
re_prod = get_topN(pred_prod, test_pdf)
re_prod.to_csv("/data_ssd/menu_norm/output/output_prod.csv", index=False)

print("3. 상호명 + 상품명\n")
K.clear_session()
K.tf.reset_default_graph()
model_all = build_model()
hist = model_all.fit(
    {'shop': shop_train, 'prod': prod_train}, y_train,
    batch_size=4000,
    epochs=2000,
    validation_data=({'shop': shop_val, 'prod': prod_val}, y_val),
    verbose=2,
    callbacks=callbacks
)

pred_all = model_all.predict({'shop': t_shop, 'prod': t_prod})
re_all = get_topN(pred_all, test_pdf)
re_all.to_csv("/data_ssd/menu_norm/output/output_all.csv", index=False)