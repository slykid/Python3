# 작성일자: 2023.07.31
# 작성자: 김길현
# 참고자료
# - https://teddylee777.github.io/huggingface/bert-kor-text-classification/
# - https://keep-steady.tistory.com/37


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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

# 데이터 로드
# data = pd.read_csv("D:/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8") # ver. Windows
# data = pd.read_csv("~/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8") # ver. mac, linux

data = pd.read_csv("D:/workspace/Python3/data/pos_menu/menu_data_jeonju.csv", sep=",", encoding="utf-8")
# data = pd.read_csv("~/workspace/Python3/data/pos_menu/menu_data_jeonju.csv", sep=",", encoding="utf-8")
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
    files=["D:/workspace/Python3/data/pos_menu/menu_data_2022.csv"],
    vocab_size=32000,
    min_frequency=5,
    limit_alphabet=6000,
    show_progress=True
)
print('train complete')

sentence = "간장양념치킨"
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

print('vocab size : %d' % tokenizer_for_load.vocab_size)

tokenized_input_for_pytorch = tokenizer_for_load('간장양념치킨', return_tensors="pt")
tokenized_input_for_tensorflow = tokenizer_for_load("간장양념치킨", return_tensors="tf")

print("Tokens (str)      : {}".format([tokenizer_for_load.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))

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

# special token 확인
tokenizer_check = BertTokenizerFast.from_pretrained(hf_model_path+'_special')

print('check special tokens : %s'%tokenizer_check.all_special_tokens[:20])

print('vocab size : %d' % tokenizer_check.vocab_size)

tokenized_input_for_pytorch = tokenizer_check("간장양념치킨", return_tensors="pt")
tokenized_input_for_tensorflow = tokenizer_check("간장양념치킨", return_tensors="tf")

print("Tokens (str)      : {}".format([tokenizer_check.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))


# load a BERT model for TensorFlow and PyTorch
model_tf = TFBertModel.from_pretrained('bert-base-cased')
model_pt = BertModel.from_pretrained('bert-base-cased')

input_tf = tokenizer_check("간장양념치킨", return_tensors="tf")
input_pt = tokenizer_check("간장양념치킨", return_tensors="pt")

# Let's compare the outputs
output_tf, output_pt = model_tf(input_tf), model_pt(**input_pt)

print('final layer output shape : %s'%(output_pt['last_hidden_state'].shape,))

# Models outputs 2 values (The value for each tokens, the pooled representation of the input sentence)
# Here we compare the output differences between PyTorch and TensorFlow.

print('\ntorch vs tf 결과차이')
for name in ["last_hidden_state", "pooler_output"]:
    print("   => {} differences: {:.5}".format(name, (output_tf[name].numpy() - output_pt[name].detach().numpy()).sum()))
