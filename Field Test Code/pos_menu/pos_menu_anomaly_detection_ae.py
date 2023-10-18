import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model

from tokenizers import BertWordPieceTokenizer

# data = pd.read_csv("data/pos_menu/pos_menu_target.csv")
data = pd.read_csv("data/pos_menu/pos_menu.csv")

# 토크나이저 초기화
tokenizer = BertWordPieceTokenizer(
    "result/pos_menu/tokenizer_model/vocab.txt",
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,  # Must be False if cased model
    lowercase=True,
    wordpieces_prefix="##",
)

data["tokens"] = data["edit_prod_nm"].apply(lambda x: tokenizer.tokenize(x))
input_sequences = [
    tokenizer.convert_tokens_to_ids(data["tokens"]) for tokens in data["tokens"]
]
max_len = max([len(seq) for seq in input_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="post")

train = data[data["menu1_nm"] != "x"].copy()
test = data[data["menu1_nm"] == "x"].copy()
train.reset_index(drop=True)
test.reset_index(drop=True)

embedding_dim = 50
hidden_units = 32

input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(tokenizer.vocab_size, embedding_dim)(input_layer)
encoder = LSTM(hidden_units, return_sequences=True)(embedding_layer)
decoder = LSTM(hidden_units, return_sequences=True)(encoder)
output_layer = TimeDistributed(Dense(tokenizer.vocab_size, activation="softmax"))(
    decoder
)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 모델 훈련
train_expanded = np.expand_dims(train, axis=-1)
test_expanded = np.expand_dims(test, axis=-1)
autoencoder.fit(train, train_expanded, epochs=10, validation_data=(test, test_expanded))

# 불용어 탐지 함수
def detect_stop_words(model, sentence):
    tokenized_sentence = tokenizer.tokenize(sentence)
    input_sequence = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    input_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [input_sequence], maxlen=max_len, padding="post"
    )

    prediction = model.predict(input_sequence_padded)[0]
    predicted_ids = np.argmax(prediction, axis=-1)

    detected_stop_words = []
    for original, predicted in zip(
        input_sequence, predicted_ids[: len(input_sequence)]
    ):
        if original == predicted:
            detected_stop_words.append(tokenizer.convert_ids_to_tokens([original])[0])

    return detected_stop_words


# 불용어 탐지 예시
for prod_nm in data["edit_prod_nm"]:
    print(f"Sentence: {prod_nm}")
    print("Detected stop words:", detect_stop_words(autoencoder, prod_nm))
