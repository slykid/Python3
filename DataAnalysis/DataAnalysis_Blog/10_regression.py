import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

matplotlib.use("MacOSX")

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


# Optimize Regression
# 1. Forward Selection
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from itertools import combinations

matplotlib.use("qtagg")

data = pd.read_csv("Dataset/housing_data/housing.data.txt", sep="\s+", header=None)
data.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# Plot for checking relationship
cols = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
sns.pairplot(data[cols], height=2.)
plt.tight_layout()
plt.show()
plt.savefig("/Users/kilhyunkim/Pictures/pairplot_housing.jpg")

# 종속 변수와 독립 변수를 정의합니다.
X = data[["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]]
y = data['MEDV']

# 상수항을 추가합니다.
X = sm.add_constant(X)

# 전진 선택법을 사용한 단계적 회귀 수행
def forward_selection(data, target, significance_level=0.05):
    initial_features = []
    best_features = list(initial_features)
    remaining_features = list(data.columns)
    remaining_features.remove('const')

    while remaining_features:
        scores_with_candidates = []
        for candidate in remaining_features:
            features = best_features + [candidate]
            X_with_candidate = sm.add_constant(data[features])
            model = sm.OLS(target, X_with_candidate).fit()
            score = model.aic
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_features == [] or best_new_score < sm.OLS(target, sm.add_constant(data[best_features])).fit().aic:
            remaining_features.remove(best_candidate)
            best_features.append(best_candidate)
        else:
            break

    return best_features

selected_features = forward_selection(X, y)
X_selected = X[selected_features]

# 최종 모델 피팅
final_model = sm.OLS(y, sm.add_constant(X_selected)).fit()
print(final_model.summary())


# 종속 변수와 독립 변수를 정의합니다.
X = data[["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]]
y = data['MEDV']

# 상수항을 추가합니다.
X = sm.add_constant(X)

# 후진 제거법을 사용한 단계적 회귀 수행
def backward_elimination(data, target, significance_level=0.05):
    features = list(data.columns)
    while len(features) > 0:
        X_with_features = sm.add_constant(data[features])
        model = sm.OLS(target, X_with_features).fit()
        max_p_value = max(model.pvalues)
        if max_p_value > significance_level:
            excluded_feature = model.pvalues.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

selected_features = backward_elimination(X, y)
X_selected = X[selected_features]

# 최종 모델 피팅
final_model = sm.OLS(y, sm.add_constant(X_selected)).fit()
print(final_model.summary())


# 종속 변수와 독립 변수를 정의합니다.
X = data[["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]]
y = data['MEDV']

# 상수항을 추가합니다.
X = sm.add_constant(X)

# 단계적 방법을 사용한 최적회귀방정식 수행
def stepwise_selection(data, target, initial_list=[], threshold_in=0.05, threshold_out=0.05):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(data.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(target, sm.add_constant(data[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True

        # backward step
        model = sm.OLS(target, sm.add_constant(data[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # all coefs except intercept
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True

        if not changed:
            break

    return included

selected_features_stepwise = stepwise_selection(X, y)
X_selected_stepwise = X[selected_features_stepwise]

# 최종 모델 피팅
final_model_stepwise = sm.OLS(y, sm.add_constant(X_selected_stepwise)).fit()

print("Stepwise Selection Model Summary:")
print(final_model_stepwise.summary())

print("Selected features by stepwise selection:", selected_features_stepwise)
