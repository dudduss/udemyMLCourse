#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:11:12 2017

@author: sampathduddu
"""

#hierarchical

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#Use dendogram to find optimal number of classifiers

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendogram")
plt.show()

# Fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0],X[y_hc == 0, 1], s=100, c ='red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0],X[y_hc == 1, 1], s=100, c ='blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0],X[y_hc == 2, 1], s=100, c ='green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0],X[y_hc == 3, 1], s=100, c ='orange', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0],X[y_hc == 4, 1], s=100, c ='yellow', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='brown', label='Centroids')
plt.show()