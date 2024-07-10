import numpy as np
import pandas as pd

# 1. Drop NA
# Case 1
data = {
    'A': [1, 5, 9],
    'B': [2, 6, 10],
    'C': [3.0, np.nan, 11.0],
    'D': [4, 8, 12]
}
df = pd.DataFrame(data)

# 결측치 탐색
na_indices = np.where(pd.isnull(df))

print("결측치의 인덱스:")
for index in range(len(na_indices[0])):
    print(f"행: {na_indices[0][index]}, 열: {na_indices[1][index]}")

# 결측치가 있는 행 제거
cleaned_df_rows = df.dropna()

# 결과 출력
print("결측치가 제거된 데이터프레임 (행 제거):")
print(cleaned_df_rows)

# 결측치가 있는 열 제거
cleaned_df_cols = df.dropna(axis=1)

# 결과 출력
print("\n결측치가 제거된 데이터프레임 (열 제거):")
print(cleaned_df_cols)


# Case 2
data = {
    'a': [1.0, 5.0, np.nan],
    'b': [2.0, 6.0, np.nan],
    'c': [3.0, np.nan, np.nan],
    'd': [np.nan, np.nan, np.nan]
}

df2 = pd.DataFrame(data)
df2.dropna(how='all')
df2.dropna(axis=1, how='all')


# 2. Replace NA
# Replace to mean value
df.fillna(df.mean(), inplace=True)
print(df)
