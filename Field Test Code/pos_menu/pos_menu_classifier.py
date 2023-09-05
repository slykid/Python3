# 작성일자: 2023.07.31
# 작성자: 김길현
# 참고자료
# - https://teddylee777.github.io/huggingface/bert-kor-text-classification/
# - https://keep-steady.tistory.com/37
# - https://dacon.io/en/codeshare/5619
# - https://github.com/pytorch/pytorch/issues/100285 : Mac MPS-파이참 사용 시 오류
# - chatgpt 사이트 질의 내용 참고

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertModel, BertConfig, AutoTokenizer


class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prod_nm = self.data.iloc[idx]["prod_nm"]
        label = self.data.iloc[idx]["label"]

        # 토큰화 처리
        tokens = self.tokenizer(
            prod_nm,  # 1개 문장
            return_tensors="pt",  # 텐서로 반환
            truncation=True,  # 잘라내기 적용
            padding="max_length",  # 패딩 적용
            add_special_tokens=True,  # 스페셜 토큰 적용
        )

        input_ids = tokens["input_ids"].squeeze(0)  # 2D -> 1D
        attention_mask = tokens["attention_mask"].squeeze(0)  # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)

        # input_ids, attention_mask, token_type_ids 이렇게 3가지 요소를 반환하도록 합니다.
        # input_ids: 토큰
        # attention_mask: 실제 단어가 존재하면 1, 패딩이면 0 (패딩은 0이 아닐 수 있습니다)
        # token_type_ids: 문장을 구분하는 id. 단일 문장인 경우에는 전부 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }, torch.tensor(label)


class CustomBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_pretrained)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dr = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, num_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = output[0]  # output["last_hidden_state"]
        x = self.dr(last_hidden_state[:, 0, :])
        x = self.fc(x)

        return x


def model_train(model, data_loader, loss_fn, optimizer, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()

    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_loss = 0
    corr = 0
    counts = 0

    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
    progress_bar = tqdm(
        data_loader, unit="batch", total=len(train_loader), mininterval=1
    )

    # mini-batch 학습을 시작합니다.
    for idx, (inputs, labels) in enumerate(progress_bar):
        # inputs, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        # 누적 Gradient를 초기화 합니다.
        optimizer.zero_grad()

        # Forward Propagation을 진행하여 결과를 얻습니다.
        output = model(**inputs)

        # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
        loss = loss_fn(output, labels)

        loss.backward()

        optimizer.step()

        _, pred = output.max(dim=1)

        corr += pred.eq(labels).sum().item()
        counts += len(labels)
        running_loss += loss.item() * labels.size(0)

    progress_bar.set_description(
        f"training loss: {running_loss / (idx + 1):.5f}, training accuracy: {corr / counts:.5f}\n"
    )
    acc = corr / len(data_loader.dataset)

    return running_loss / len(data_loader.dataset), acc


def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        corr = 0
        running_loss = 0

        for inputs, labels in data_loader:

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            output = model(**inputs)

            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(labels)).item()
            running_loss += loss_fn(output, labels).item() * labels.size(0)

        acc = corr / len(data_loader.dataset)

        return running_loss / len(data_loader.dataset), acc


# 변수 설정
CHECKPOINT_NAME = "Models/bert-kor-base"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

# 데이터 로드
train = pd.read_csv("../../data/pos_menu/train.csv")
test = pd.read_csv("../../data/pos_menu/test.csv")
df_label = pd.read_csv("../../data/pos_menu/label.csv")

num_class = len(df_label.num)


# 최소 1회만 수행함
# # 신규 토큰 추출
# tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
#
# corpus_file = ["data/pos_menu/menu_corpus.csv"]  # data path
# vocab_size = 32000
# limit_alphabet = 6000
# output_path = "result/pos_menu/bert-kor-base"
# min_frequency = 5
#
# # Then train it!
# tokenizer.train(
#     files=corpus_file,
#     vocab_size=vocab_size,
#     min_frequency=min_frequency,  # 단어의 최소 발생 빈도, 5
#     limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
#     show_progress=True,
# )
# print("train complete")
#
# tokenizer.save_model(output_path)
#
# # 기존 tokenizer에 신규 토큰 추가
# tokenizer = BertTokenizerFast.from_pretrained("Models/bert-kor-base")
# new_tokenizer = BertTokenizerFast.from_pretrained("result/pos_menu/bert-kor-base")
#
# new_token = set(new_tokenizer.get_vocab().keys()) - set(tokenizer.get_vocab().keys())
# tokenizer.add_tokens(list(new_token))
# tokenizer.save_pretrained("Models/bert-kor-base")


# train, test 데이터셋 생성

CHECKPOINT_NAME = "Models/bert-kor-base"
train_data = TokenDataset(train, CHECKPOINT_NAME)
test_data = TokenDataset(test, CHECKPOINT_NAME)

# DataLoader로 이전에 생성한 Dataset를 지정하여, batch 구성, shuffle, num_workers 등을 설정합니다.
# - num_worker에 0 이외 숫자부여 시, 병렬화 프로세스 에 피클 데이터를 전달 과정 상 문제로 AttributeError: Can't get attribute 'TokenDataset' on <module '__main__' (built-in)> 에러 발생함
# - 참고자료: https://www.reddit.com/r/MLQuestions/comments/n9iu83/pytorch_lightning_bert_attributeerror_cant_get/
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=0)

# 아래는 확인용이니 실행안해도 무방함
# # - 실행 시 반드시 재실행해야됨 CUDA OOM 날 수 있음
# inputs, labels = next(iter(train_loader))
#
# # 데이터셋을 device 설정
# inputs = {k: v.to(device) for k, v in inputs.items()}
# labels.to(device)
#
# # 생성된 inputs의 key 값 출력
# inputs.keys()
#
# # key 별 shape 확인
# inputs["input_ids"].shape
# inputs["attention_mask"].shape
# inputs["token_type_ids"].shape
#
# # labels 출력
# labels
#
# # 테스트
# model_bert = BertModel.from_pretrained(CHECKPOINT_NAME).to(device)
# model_bert.resize_token_embeddings(len(tokenizer))
#
# output = model_bert(**inputs)
# output.keys()
#
# output["last_hidden_state"].shape, output["pooler_output"].shape
# last_hidden_state = output["last_hidden_state"]
# print(last_hidden_state.shape)
# print(last_hidden_state[:, 0, :])
#
# pooler_output = output["pooler_output"]
# print(pooler_output.shape)
# print(pooler_output)
#
# fc = nn.Linear(768, 2)
# fc.to(device)
# fc_output = fc(last_hidden_state[:, 0, :])
# print(fc_output.shape)
# print(fc_output.argmax(dim=1))
#
# del inputs
# del model_bert
# del output
# del fc
# del fc_output


# modeling & training

CHECKPOINT_NAME = "Models/bert-kor-base"

tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT_NAME)
bert = CustomBertModel(CHECKPOINT_NAME)
bert.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert.parameters(), lr=1e-5)

n_epochs = 10

model_name = "bert-kor-base"
min_loss = np.inf

for epoch in range(n_epochs):
    train_loss, train_acc = model_train(bert, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = model_evaluate(bert, test_loader, loss_fn, device)

    if val_loss < min_loss:
        print(
            f"[INFO] val_loss has been improved from {min_loss: .5f} to {val_loss: .5f}. Saving Model!\n"
        )
        min_loss = val_loss
        torch.save(bert.state_dict(), f"{model_name}.pth")

    print(
        f"epoch {epoch+1: 02d}, loss: {train_loss: .5f}, acc: {train_acc: .5f}, val_loss: {val_loss: .5f}, val_acc: {val_acc: .5f}\n"
    )
