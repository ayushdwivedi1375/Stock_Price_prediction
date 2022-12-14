# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 15:11:07 2022

@author: hp
"""

import pandas as pd
import numpy as np
import math
from sklearn import svm
from matplotlib import pyplot as plt
import yfinance as yf
d1 = yf.download("MRVL", start="2007-01-01", end="2017-04-30")

d1['Open-Close'] = d1.Open - d1.Close
d1['High-Low'] = d1.High - d1.Low
X = d1[['Open-Close', 'High-Low']]
y = pd.Series(np.where(d1['Close'].shift(-1) > d1['Close'], 1, 0))

n_train = math.floor(0.6 * X.shape[0])
n_test = math.ceil(0.4 * X.shape[0])
X1_train = X[:n_train]
y_train =y[:n_train]
X1_test = X[n_train:]
y_test = y[n_train:]

Classifier=svm.SVC(kernel='linear',C=2.0,gamma='auto')
Classifier.fit(X1_train, y_train)
y_predict=Classifier.predict(X1_test)
print("For C=2\n")
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))






