from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        super().__init__()

    def _train(self, x, y, config):
        # 모델 모드 전환 : train 모드
        self.model.train()

        # shuffle 수행
        indices = torch.randperm(x.size(0), device=x.device)  # shuffle index 생성

        # x, y 모두 동일하게 셔플링 수행
        # index_select(입력 변수, dim=차원, index=추출 인덱스 번호)
        # - dim = 0 이면 행, dim = 1 이면 열
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.criterion(y_hat_i, y_i.squeeze())  # cross entropy loss

            self.optimizer.zero_grad() # Gradient Initialization
            loss_i.backward()

            self.optimizer.step()  # weight update

            if config.verbose >= 2:
                print("Train iteration(%d/%d): loss = %.4e" % (i+1, len(x), float(loss_i)))

            total_loss += float(loss_i)  # loss_i 가 텐서형태이기 때문에 float() 형 변환을 안할 경우
                                         # total_loss 역시 텐서로 저장되며, 이는 메모리 공간 부족을 유발함

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # 모델 모드 전환 : evaluate 모드
        self.model.eval()

        with torch.no_grad():
            indice = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indice).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indice).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.criterion(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

        return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) : train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" %
                  (epoch + 1, config.n_epochs, train_loss, valid_loss, lowest_loss))

            self.model.load_state_dict(best_model)