# 1. K-Means
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from matplotlib import pyplot as plt

x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c='white', marker='o', edgecolors='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
pred = kmeans.fit_predict(x)

# 결과 시각화
plt.scatter(x[pred == 0, 0],
            x[pred == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolors='black',
            label='cluster 1')

plt.scatter(x[pred == 1, 0],
            x[pred == 1, 1],
            s=50, c='orange',
            marker='s', edgecolors='black',
            label='cluster 2')

plt.scatter(x[pred == 2, 0],
            x[pred == 2, 1],
            s=50, c='lightblue',
            marker='s', edgecolors='black',
            label='cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolors='black',
            label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

# K Means++
kmeans_plus = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, tol=1e-04, random_state=0)
pred_plus = kmeans_plus.fit_predict(x)

plt.scatter(x[pred_plus == 0, 0],
            x[pred_plus == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolors='black',
            label='cluster 1')

plt.scatter(x[pred_plus == 1, 0],
            x[pred_plus == 1, 1],
            s=50, c='orange',
            marker='s', edgecolors='black',
            label='cluster 2')

# plt.scatter(x[pred_plus == 2, 0],
#             x[pred_plus == 2, 1],
#             s=50, c='lightblue',
#             marker='s', edgecolors='black',
#             label='cluster 3')

plt.scatter(kmeans_plus.cluster_centers_[:, 0],
            kmeans_plus.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolors='black',
            label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()





print('왜곡 : %.2f' % kmeans_plus.inertia_)

distortions = []
for i in range(1, 11):
    kmeans_plus = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans_plus.fit(x)
    distortions.append(kmeans_plus.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(pred_plus)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(x, pred_plus, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[pred_plus == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_upper + y_ax_lower) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(c_silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.tight_layout()
plt.show()


# 계층 군집
import pandas as pd
import numpy as np

np.random.seed(12)
variables = ['x', 'y', 'z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

x = np.random.random_sample([5, 3])*10
df = pd.DataFrame(x, columns=variables, index=labels)
df

from scipy.spatial.distance import pdist, squareform

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
row_dist

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
help(linkage)

# 잘못된 방식
# - 아래 코드에서 row_dist 는 squareform() 을 통해 생성된 거리행렬이다.
# - 잘못된 이유는 linkage() 의 입력값으로 squareform() 으로 생성된 거리행렬을
#   사용할 수 없기 때문이다.
# row_clusters = linkage(row_dist, method='complete', metric='euclidean')

# 올바른 방식 1
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

# 올바른 방식 2
row_clusters = linkage(df.values, method='complete', metric='euclidean')

pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2', 'distance', 'item no.'],
             index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])

row_dendrogram = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()

# sklearn hierarchy clustering
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(x)
print("클래스 레이블 : %s" % labels)
