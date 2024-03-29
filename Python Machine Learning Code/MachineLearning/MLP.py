import numpy as np
import sys

class NeuralNetMLP(object):
    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        """
        :param n_hidden: 은닉 유닛 수 / int
        :param l2: L2 규제의 람다 값 / float
        :param epochs: 훈련 세트 반복 횟수 / int
        :param eta: 학습률 / float
        :param shuffle: 훈련 세트 셔플 여부 / bool
        :param minibatch_size: 미니 배치 샘플 개수 / int
        :param seed: 난수 초기 값 / int
        """
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """
        :param y: 타겟 / 배열 ( 크기 = n_samples )
        :param n_classes: 클래수 개수 / int
        :return: ont-hot vec. / 배열 ( 크기 = (n_samples, n_labels) )
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.

        return onehot.T

    # 시그모이드 함수
    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    # 정방향 전파 (Feed-Forward Progagation)
    def _forward(self, x):
        z_h = np.dot(x, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h)

        z_out = np.dot(a_h, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    # 비용 함수
    def _compute_cost(self, y_enc, output):
        """
        :param y_enc: one-hot 인코딩 된 클래스 레이블 / 배열 ( 크기 = (n_samples, n_labels) )
        :param output: 출력층의 활성화 출력 / 배열 ( 크기 = [n_samples, n_output_units]
        :return: cost (규제가 포함된 비용) / float
        """
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term

        return cost

    # 결과 예측 함수
    def _predict(self, x):
        """
        :param x: 원본 특성의 입력층 / 배열 ( 크기 = [n_samples, n_features] )
        :return: y_pred (예측된 클래스 레이블) / 배열 ( 크기 = [n_samples] )
        """
        z_h, a_h, z_out, a_out = self._forward(x)
        y_pred = np.argmax(z_out, axis=1)

        return y_pred

    # 학습 함수
    def _fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: 원본 특성의 입력층 / 배열 ( 크기 = [n_samples, n_features] )
        :param y_train: 타겟 클래스 레이블 / 배열 ( 크기 = [n_samples] )
        :param x_valid: 훈련 중 검증용 데이터 / 배열 ( 크기 = [n_samples, n_features] )
        :param y_valid: 훈련 중 검증용 데이터에 대한 타겟 클래스 레이블 / 배열 ( 크기 = [n_samples] )
        :return: self
        """
        n_output = np.unique(y_train).shape[0]
        n_features = x_train.shape[1]

        # 가중치 초기화
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # 은닉층 -> 출력층 사이의 가중치
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost' : [], 'train_acc' : [], 'valid_acc' : []}

        y_train_enc = self._onehot(y_train, n_output)

        # 훈련
        for i in range(self.epochs):

            # 미니배치 생성
            indices = np.arange(x_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx : (start_idx + self.minibatch_size)]

                # 정방향 계산
                z_h, a_h, z_out, a_out = self._forward(x_train[batch_idx])

                # 역전파 계산
                sigma_out = a_out - y_train_enc[batch_idx]
                sigmoid_derivative_h = a_h * (1. - a_h)

                sigma_h = (np.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h)

                ## 그레디언트 계산
                grad_w_h = np.dot(x_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)


                ## 규제 및 가중치 업데이트
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = grad_w_out + self.l2 * self.w_out
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            # 평가
            z_h, a_h, z_out, a_out = self._forward(x_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            y_train_pred = self._predict(x_train)
            y_valid_pred = self._predict(x_valid)

            train_acc = (np.sum(y_train == y_train_pred)).astype(float) / x_train.shape[0]
            valid_acc = (np.sum(y_valid == y_valid_pred)).astype(float) / x_valid.shape[0]

            sys.stderr.write("%0*d/%d | 비용: %.2f | 훈련/검증 정확도 : %.2f%%/%.2f%%\n" % (epoch_strlen, i+1, self.epochs, cost, train_acc*100, valid_acc*100))

            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self
