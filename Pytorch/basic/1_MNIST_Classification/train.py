import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from dataloader import load_mnist

def define_argparser():
    # 필요한 인자들 선언
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)  # 모델 및 가중치 저장할 파일명
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)  # 학습시 사용할 장치(CPU, GPU)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config

def main(config):
    # 실행 장치 설정
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # 데이터 로드
    x, y = load_mnist(is_train=True)
    x = x.view(x.size(0), -1)  # flatten 된 형태

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle & Random Split
    indices = torch.randperm(x.size(0))
    x = torch.index_select(x, dim=0, index=indices).to(device).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).to(device).split([train_cnt, valid_cnt], dim=0)
    # split 되는 형태: [(train_cnt, 784) , (valid_cnt, 784)]

    print("Train : ", x[0].shape, y[0].shape)
    print("Valid : ", x[1].shape, y[1].shape)

    # load model, optimizer, criterion
    model = ImageClassifier(28**2, 10).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion)

    trainer.train((x[0], y[0]), (x[1], y[1]), config)
    
    # 베스트 모델 및 가중치 저장
    torch.save({
        'model' : trainer.model.state_dict(),
        'config' : config,
    }, config.model_fn)

if __name__ == '__main__':
    # 문제 정의
    # 입력 784 차원 (28 x 28) 크기의 이미지를 입력으로 받아 10 차원으로 출력

    # 실행 예시(CLI) : python train.py --model_fn model.pth -- gpu_id -1 --batch_size 512 --n_epochs 20 --verbose 2
    config = define_argparser()
    main(config)











