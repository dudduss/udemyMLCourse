#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:15:19 2017

@author: sampathduddu
"""

#kmeans clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('curveball.csv')



dataset_orig = dataset.iloc[:,:20]
dataset = dataset_orig.iloc[:,6:]

X = dataset.values
 


# Find optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,21), wcss)
plt.show()

#Applying kmeans to the data set with the correct number of clusters
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X)



# Visualizing the clusters
#plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1], s=100, c ='red', label = 'Cluster 1')
#plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1], s=100, c ='blue', label = 'Cluster 2')
#plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1], s=100, c ='green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3, 1], s=100, c ='orange', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4, 1], s=100, c ='yellow', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='brown', label='Centroids')
#plt.show()
    
    