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

data = pd.read_csv("D:/workspace/Python3/data/pos_menu/menu_data_2022.csv", sep=",", encoding="utf-8")
data

# wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
# wordpiece_tokenizer.train(
#     files=["D:/workspace/Python3/data/pos_menu/menu_data_2022.csv"],
#     vocab_size=10000,
# )
# wordpiece_tokenizer.save_model("result/pos_menu/")

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

corpus = Korpora.load("nsmc")

tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
model_bert = BertModel.from_pretrained("kykim/bert-kor-base")





