from sklearn.datasets import california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

housing = california_housing.fetch_california_housing(".")

features = housing.feature_names
X = housing.data
y = housing.target
#pd_data = np.vstack([features,X])
#print pd_data
df = pd.DataFrame(data=X, columns=features)
print df.info()
#df.hist(bins=50, figsize=(20,15))
#plt.show()

#corr = df.corr()
#print corr

print df.describe()

pipeline = Pipeline([('std_scaler', StandardScaler())])

housing_tr = pipeline.fit_transform(df)

print pd.DataFrame(housing_tr, columns=features).describe()
