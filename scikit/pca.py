import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X = np.random.rand(10,3)
X_centered = X - X.mean(axis=0)

U,s,V = np.linalg.svd(X_centered)

c1 = V.T[:,0] #pca c1
c2 = V.T[:,1] #pca c2
X2D = X_centered.dot(V.T[:,:2])
print X2D

pca = PCA(n_components=2)

X2D = pca.fit_transform(X)
print X2D
print pca.explained_variance_ratio_

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print X_reduced
print pca.explained_variance_ratio_