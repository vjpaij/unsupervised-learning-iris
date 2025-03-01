import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load the iris dataset
iris_df = sns.load_dataset('iris')

sns.set_style = 'darkgrid'
# sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species')
# plt.show()

numeric_cols = iris_df.select_dtypes(include=[np.number]).columns
X = iris_df[numeric_cols]

#Using K-Means method
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
print(model.cluster_centers_)
preds = model.predict(X)
print(preds)

# sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds)
# centers_x, centers_y = model.cluster_centers_[:, 0], model.cluster_centers_[:, 2]
# plt.plot(centers_x, centers_y, 'xb')
# plt.show()

#variance is obtained by inertia. Lesser the variance, lesser is the spread.
print(model.inertia_)

#In real world scenarios, we wouldnt have pre determined clusters
#We can plot "No of Clusters" vs "Inertia" to pick the right number
#the elbow where the curve starts to flatten, is the cluster we may choose. 6 in this example.
options = range(2, 11)
inertias = []

for n_cluster in options:
    model = KMeans(n_clusters=n_cluster, random_state=42).fit(X)
    inertias.append(model.inertia_)

# plt.title("Clusters vs Inertia")
# plt.plot(options, inertias, '-o')
# plt.xlabel('Clusters (K)')
# plt.ylabel('Inertia')
# plt.show()

#Using DBSCAN method
from sklearn.cluster import DBSCAN

model_db = DBSCAN(eps=1.1, min_samples=4)
model_db.fit(X)

#DBSCAN doesn't have prediction steps. It directly assigns labels to all inputs
print(model_db.labels_)

# sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=model_db.labels_)
# plt.show()

# Principal Component Analysis - Dimension Reductionality. 
# If we have 100 columns, it is difficult to build a model and takes more resources and time. Reducing the columns to say 5 columns
# and try to retain as much information as possible.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(iris_df[numeric_cols])
transformed = pca.transform(iris_df[numeric_cols])

sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df['species'])
plt.show()


