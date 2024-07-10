import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('qtagg')

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris.data, iris.target],
                  columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
df.head()

train_df = df[['sepal length', 'sepal width', 'petal length', 'petal width']]

tsne_np = TSNE(n_components=2).fit_transform(train_df)

tsne_df = pd.DataFrame(tsne_np, columns=['component 0', 'component 1'])
tsne_df

tsne_df['target'] = df['target']
tsne_df_0 = tsne_df[tsne_df['target'] == 0]
tsne_df_1 = tsne_df[tsne_df['target'] == 1]
tsne_df_2 = tsne_df[tsne_df['target'] == 2]

plt.scatter(tsne_df_0['component 0'], tsne_df_0['component 1'], color='pink', label='setosa')
plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], color='purple', label='versicolor')
plt.scatter(tsne_df_2['component 0'], tsne_df_2['component 1'], color='yellow', label='virginica')

plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()
plt.show()