import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_preprocessing.text import Tokenizer

from gensim.models import word2vec

data = pd.read_csv("data/Q11002.csv")
word_vec = word2vec.Word2Vec(data["prod_nm"].tolist(), min_count=1)
word_vec.build_vocab(data["prod_nm"].tolist())


model = word2vec.Word2Vec([data["prod_nm"]],       # 리스트 형태의 데이터
                 sg=1,         # 0: CBOW, 1: Skip-gram
                 size=100,     # 벡터 크기
                 window=3,     # 고려할 앞뒤 폭(앞뒤 3단어)
                 min_count=3,  # 사용할 단어의 최소 빈도(3회 이하 단어 무시)
                 workers=4)    # 동시에 처리할 작업 수(코어 수와 비슷하게 설정)