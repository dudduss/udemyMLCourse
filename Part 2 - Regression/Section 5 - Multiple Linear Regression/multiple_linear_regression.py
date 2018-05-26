#Multiple Linear Regression
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 21:01:27 2016

@author: sampathduddu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting test set results
Y_pred = regressor.predict(X_test)

#Building optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(np.ones((50, 1)).astype(int), X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]] #only contains variables that have high impact on dependent variable
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #OLS is ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]] #only contains variables that have high impact on dependent variable
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #OLS is ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]] #only contains variables that have high impact on dependent variable
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #OLS is ordinary least squares
regressor_OLS.summary()
 
X_opt = X[:, [0,3,5]] #only contains variables that have high impact on dependent variable
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #OLS is ordinary least squares
regressor_OLS.summary()
 
X_opt = X[:, [0,3]] #only contains variables that have high impact on dependent variable
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #OLS is ordinary least squares
regressor_OLS.summary() # only one dependent variable












