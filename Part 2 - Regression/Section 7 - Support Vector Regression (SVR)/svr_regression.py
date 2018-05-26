#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:54:29 2017

@author: sampathduddu
"""

#SVR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#SVR Regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf is gaussian curve
regressor.fit(X, Y)

# Plotting and Predicting
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X))