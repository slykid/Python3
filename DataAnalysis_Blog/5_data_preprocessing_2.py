from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

print("변경 전: ")
print(x)

scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x)

print("변경 후")
print(x_norm)
