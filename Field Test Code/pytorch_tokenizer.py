# 작성일자: 2023.07.31
# 작성자: 김길현
# 참고자료
# - https://teddylee777.github.io/huggingface/bert-kor-text-classification/
# - https://keep-steady.tistory.com/37
# - https://dacon.io/en/codeshare/5619
# - chatgpt 사이트 질의 내용 참고

import os
import re, collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast, BertModel, BertConfig

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


# 변수 설정
num_class = 806  # pos menu3_nm 기준

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

# 데이터 로드
# data = pd.read_csv("data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8")
data = pd.read_csv("data/pos_menu/menu_data_jeonju.csv", sep=",", encoding="utf-8")
data

# 학습용 테스트용 분리
train = data.loc[-data.isna().any(axis=1)]
test = data.loc[data.isna().any(axis=1)]

# 문자 길이 컬럼 추가
train["label"] = 0
label = pd.unique(train.menu3_nm.apply(lambda x: str(x))).tolist()
label.sort()
label = [x for x in label if x != 'nan']
label_num = {word : num for num, word in enumerate(label)}
train["label"] = train["menu3_nm"].apply(lambda x: label_num.get(x))

train["length"], test["length"] = 0, 0
train["length"] = train["prod_nm"].apply(lambda x: len(str(x)))
test["length"] = test["prod_nm"].apply(lambda x: len(str(x)))

train["id"], test["id"] = 0, 0
train["id"] = [x for x in range(1, len(train.prod_nm) + 1)]
test["id"] = [x for x in range(1, len(test.prod_nm) + 1)]

train = train[["id", "prod_nm", "label", "length"]]
test = test[["id", "prod_nm", "length"]]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# tokenizer
tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
tokenizer.get_vocab()

# tokenizer에 special token 추가
user_defined_symbols = ['[BOS]','[EOS]','[UNK0]','[UNK1]','[UNK2]','[UNK3]','[UNK4]','[UNK5]','[UNK6]','[UNK7]','[UNK8]','[UNK9]']
unused_token_num = 200
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
user_defined_symbols = user_defined_symbols + unused_list

tokenizer.all_special_tokens
special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.all_special_tokens



CHECKPOINT_NAME = 'kykim/bert-kor-base'
tokenizer_pretrained = CHECKPOINT_NAME

# train, test 데이터셋 생성
train_data = TokenDataset(train, tokenizer_pretrained)
test_data = TokenDataset(test, tokenizer_pretrained)

# DataLoader로 이전에 생성한 Dataset를 지정하여, batch 구성, shuffle, num_workers 등을 설정합니다.
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=2)

inputs, labels = next(iter(train_loader))

# 데이터셋을 device 설정
inputs = {k: v.to(device) for k, v in inputs.items()}
labels.to(device)

# 생성된 inputs의 key 값 출력
inputs.keys()

# key 별 shape 확인
inputs['input_ids'].shape, inputs['attention_mask'].shape, inputs['token_type_ids'].shape

config = BertConfig.from_pretrained(CHECKPOINT_NAME)
config

# labels 출력
labels

# 모델 생성
model_bert = BertModel.from_pretrained(CHECKPOINT_NAME).to(device)
model_bert

# loss 정의: CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저 정의: bert.paramters()와 learning_rate 설정
optimizer = optim.Adam(bert.parameters(), lr=1e-5)