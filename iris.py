import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load the iris dataset
iris_df = sns.load_dataset('iris')

sns.set_style = 'darkgrid'
#sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species')
# plt.show()

numeric_cols = iris_df.select_dtypes(include=[np.number]).columns
X = iris_df[numeric_cols]

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
print(model.cluster_centers_)
preds = model.predict(X)
print(preds)

# sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds)
# centers_x, centers_y = model.cluster_centers_[:, 0], model.cluster_centers_[:, 2]
# plt.plot(centers_x, centers_y, 'xb')
# #plt.show()

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

plt.title("Clusters vs Inertia")
plt.plot(options, inertias, '-o')
plt.xlabel('Clusters (K)')
plt.ylabel('Inertia')
plt.show()



