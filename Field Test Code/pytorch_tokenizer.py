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

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertModel, TFBertModel

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

# 데이터 로드
# data = pd.read_csv("D:/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8") # ver. Windows
# data = pd.read_csv("~/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8") # ver. mac, linux

# data = pd.read_csv("D:/workspace/Python3/data/pos_menu/menu_data_jeonju.csv", sep=",", encoding="utf-8")
data = pd.read_csv("~/workspace/Python3/data/pos_menu/menu_data_jeonju.csv", sep=",", encoding="utf-8")
data

# 학습용 테스트용 분리
train = data.loc[-data.isna().any(axis=1)]
test = data.loc[data.isna().any(axis=1)]

# 문자 길이 컬럼 추가
train["length"] = train["prod_nm"].apply(lambda x: len(str(x)))
test["length"] = test["prod_nm"].apply(lambda x: len(str(x)))

# tokenizer
tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
tokenizer.train(
    # files=["D:/workspace/Python3/data/pos_menu/menu_data_2022.csv"],
    files=["/Users/slykid/workspace/Python3/data/pos_menu/menu_data_2022.csv"],
    vocab_size=32000,
    min_frequency=5,
    limit_alphabet=6000,
    show_progress=True
)
print('train complete')

user_defined_symbols = ['[BOS]','[EOS]','[UNK0]','[UNK1]','[UNK2]','[UNK3]','[UNK4]','[UNK5]','[UNK6]','[UNK7]','[UNK8]','[UNK9]']
unused_token_num = 200
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
user_defined_symbols = user_defined_symbols + unused_list

special_tokens_dict = {'additional_special_tokens': user_defined_symbols}

sentence = "광어한판 10pcs"
output = tokenizer.encode(sentence)

print(sentence)
print('=>idx   : %s'%output.ids)
print('=>tokens: %s'%output.tokens)
print('=>offset: %s'%output.offsets)
print('=>decode: %s\n'%tokenizer.decode(output.ids))

hf_model_path='result/pos_menu/tokenizer_model'
if not os.path.isdir(hf_model_path):
    os.mkdir(hf_model_path)
tokenizer.save_model(hf_model_path)

# Tokenizer 테스트
tokenizer_for_load = BertTokenizerFast.from_pretrained(hf_model_path,
                                                       strip_accents=False,  # Must be False if cased model
                                                       lowercase=False)  # 로드

# vocab check
tokenizer_for_load.get_vocab()

# special token check
tokenizer_for_load.all_special_tokens # 추가하기 전 기본적인 special token

# tokenizer에 special token 추가
user_defined_symbols = ['[BOS]','[EOS]','[UNK0]','[UNK1]','[UNK2]','[UNK3]','[UNK4]','[UNK5]','[UNK6]','[UNK7]','[UNK8]','[UNK9]']
unused_token_num = 200
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
user_defined_symbols = user_defined_symbols + unused_list

special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
tokenizer_for_load.add_special_tokens(special_tokens_dict)

# check tokenizer vocab with special tokens
print('check special tokens : %s'%tokenizer_for_load.all_special_tokens[:20])

# save tokenizer model with special tokens
tokenizer_for_load.save_pretrained(hf_model_path+'_special')

