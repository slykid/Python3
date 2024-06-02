import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

matplotlib.use("qtagg")

# 데이터 생성
np.random.seed(42)
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



# 데이터 생성
np.random.seed(42)
x1 = 2 * np.random.rand(100, 1)
x2 = 3 * np.random.rand(100, 1)
y = 3 + 4 * x1 + 5 * x2 + np.random.randn(100, 1)

# 데이터 프레임 생성
data = np.hstack((x1, x2))
df = pd.DataFrame(data, columns=['x1', 'x2'])
df['y'] = y

# 독립변수에 상수항 추가
X = sm.add_constant(df[['x1', 'x2']])
y = df['y']

# 모델 적합
model = sm.OLS(y, X).fit()

# 모델 요약 정보 출력
print(model.summary())

# 잔차 계산
residuals = model.resid

# 잔차 그래프 그리기
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')
plt.show()
plt.savefig("/Users/slykid/Pictures/regression1-residuals.jpg")

# 잔차의 정규성 검토 - Q-Q plot
sm.qqplot(residuals, line='45')
plt.title('Q-Q plot of residuals')
plt.show()
plt.savefig("/Users/slykid/Pictures/regression1-qq.jpg")

# 잔차의 히스토그램
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of residuals')
plt.show()
plt.savefig("/Users/slykid/Pictures/regression1-hist.jpg")

# 잔차의 통계 요약
print(f'Residuals mean: {np.mean(residuals):.4f}')
print(f'Residuals variance: {np.var(residuals):.4f}')