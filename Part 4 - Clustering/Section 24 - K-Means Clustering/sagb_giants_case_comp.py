# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 21:04:08 2017

@author: sampathduddu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('curveball_labeled.csv')
dataset = dataset[dataset['pitch_type'] == 'CB']
dataset_orig = dataset.iloc[:,:20]
dataset = dataset_orig.iloc[:,6:]

X = dataset.iloc[:, 0:14].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
s = pca.components_

principal_dataset = np.dot(dataset.values, np.transpose(s))


# Find optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(principal_dataset)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss)
plt.show()

#Applying kmeans to the data set with the correct number of clusters
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(principal_dataset)

# Visualizing the clusters
plt.scatter(principal_dataset[y_kmeans == 0, 0],principal_dataset[y_kmeans == 0, 1], s=100, c ='red', label = 'C1')
plt.scatter(principal_dataset[y_kmeans == 1, 0],principal_dataset[y_kmeans == 1, 1], s=100, c ='blue', label = 'C2')
plt.scatter(principal_dataset[y_kmeans == 2, 0],principal_dataset[y_kmeans == 2, 1], s=100, c ='green', label = 'C3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='')
plt.title('Clusters of Curveball Pitches')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
