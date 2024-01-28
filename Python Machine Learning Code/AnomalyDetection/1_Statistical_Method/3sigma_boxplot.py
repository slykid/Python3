import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

# 실습 데이터 생성
df = pd.DataFrame({
    "name": ["KATE", "LOUISE", "JANE", "JASON", "TOM", "JACK"],
    "weight": ["59", "61", "55", "66", "52", "110"],
    "height": ["120", "123", "115", "145", "64", "20"]
})

print(df)
print(df.info())

# 숫자형 데이터 변경
df["weight"] = df["weight"].astype(int)
df["height"] = df["height"].astype(int)

print(df.info())  # 변경여부 확인

# UCL, LCL 설정
df["UCL_W"] = df["weight"].mean() + 2 * df["weight"].std()
df["LCL_W"] = df["weight"].mean() - 2 * df["weight"].std()
df["UCL_H"] = df["height"].mean() + 2 * df["height"].std()
df["LCL_H"] = df["height"].mean() - 2 * df["height"].std()

print(df)  # 데이터 확인

# 시각화
plt.style.use(["dark_background"])

# weight plot
sns.scatterplot(x=df["name"], y=df["weight"]);
plt.axhline(y=df["UCL_W"][0], color='r', linewidth=2)
plt.axhline(y=df["LCL_W"][0], color='r', linewidth=2)
plt.gcf().set_size_inches(15, 5)

# height plot
sns.scatterplot(x=df["name"], y=df["height"])
plt.axhline(y=df["UCL_H"][0], color='r', linewidth=2)
plt.axhline(y=df["LCL_H"][0], color='r', linewidth=2)
plt.gcf().set_size_inches(15, 5)

