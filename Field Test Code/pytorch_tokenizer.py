# 작성일자: 2023.07.31
# 작성자: 김길현
# 참고자료: https://teddylee777.github.io/huggingface/bert-kor-text-classification/

import os
import re, collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast, BertModel
from Korpora import Korpora

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        # sentence, label 컬럼으로 구성된 데이터프레임 전달
        self.data = dataframe
        # Huggingface 토크나이저 생성
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['prod_nm']
        label = self.data.iloc[idx]['label']

        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,  # 1개 문장
            return_tensors='pt',  # 텐서로 반환
            truncation=True,  # 잘라내기 적용
            padding='max_length',  # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)  # 2D -> 1D
        attention_mask = tokens['attention_mask'].squeeze(0)  # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)

        # input_ids, attention_mask, token_type_ids 이렇게 3가지 요소를 반환하도록 합니다.
        # input_ids: 토큰
        # attention_mask: 실제 단어가 존재하면 1, 패딩이면 0 (패딩은 0이 아닐 수 있습니다)
        # token_type_ids: 문장을 구분하는 id. 단일 문장인 경우에는 전부 0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

# 데이터 로드
# data = pd.read_csv("D:/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8") # ver. Windows
# data = pd.read_csv("~/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8") # ver. mac, linux
data = pd.read_csv("~/workspace/Python3/data/pos_menu/menu_data_jeonju.csv", sep=",", encoding="utf-8")
data

# 학습용 테스트용 분리
train = data.loc[-data.isna().any(axis=1)]
test = data.loc[data.isna().any(axis=1)]

# 문자 길이 컬럼 추가
train["length"] = train["prod_nm"].apply(lambda x: len(str(x)))
test["length"] = test["prod_nm"].apply(lambda x: len(str(x)))

# wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
# wordpiece_tokenizer.train(
#     files=["D:/workspace/Python3/data/pos_menu/menu_data_2022.csv"],
#     vocab_size=10000,
# )
# wordpiece_tokenizer.save_model("result/pos_menu/")

corpus = Korpora.load("nsmc")

tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
model_bert = BertModel.from_pretrained("kykim/bert-kor-base")




