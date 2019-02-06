# -*- coding: utf-8 -*-
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Importing the dataset
dataset = pd.read_csv('parameters dataset.csv')
X = dataset.iloc[:, 2:10].values
y = dataset.iloc[:, 12].values
print(X)
print(y)
dataset.plot()

"""f= plt.subplots(figsize=(11,11))
sn.heatmap(dataset.corr(),annot=True,fmt='.1f',color='green')
pd.plotting.scatter_matrix(dataset.loc[0:,dataset.columns],c=['red','blue'],
                           alpha=0.5,figsize=[25,25],diagonal='hist',s=200,marker='.',
                           edgecolor='black')"""
plt.show()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

"""
# Avoiding the Dummy Variable Trap
X = X[:, 1:]"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# Evaluation of metrics
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 