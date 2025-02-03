import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load the iris dataset
iris_df = sns.load_dataset('iris')

sns.set_style = 'darkgrid'
sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species')
# plt.show()

numeric_cols = iris_df.select_dtypes(include=[np.number]).columns
X = iris_df[numeric_cols]

