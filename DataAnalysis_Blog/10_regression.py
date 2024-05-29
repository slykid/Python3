import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

matplotlib.use("qtagg")

# 데이터 생성
x = 2 * np.random.rand(100, 1)
y = 3 + x + np.random.randn(100, 1)

# 선형 회귀 모델 생성 및 적합
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)

# 예측 결과와 실제 데이터 시각화
plt.plot(x, y_pred, "r-", linewidth=2, label="Predictions")
plt.plot(x, y, "b.", label="Actual data")
plt.xlabel("x(특성)")
plt.ylabel("y(타깃)")
plt.legend()
plt.show()




