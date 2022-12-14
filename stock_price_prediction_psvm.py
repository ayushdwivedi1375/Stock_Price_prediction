# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:33:04 2022

@author: hp
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn
import yfinance as yf
data = yf.download("MRVL", start="2015-01-01", end="2017-04-30")
data['Open-Close'] = data.Open - data.Close
data['High-Low'] = data.High - data.Low
X = data[['Open-Close', 'High-Low']]
y = pd.Series(np.where(data['Close'].shift(-1) > data['Close'], 1, 0))
split_percentage = 0.8
split = int(split_percentage*len(data))
  
# Train data set
X_train = X[:split]
y_train = y[:split]
  
# Test data set
X_test = X[split:]
y_test = y[split:]

B = []
lab = list(np.array(y_train))
for i in range(len(y_train)):
    A = []
    for j in range(len(y_train)):
        if i == j:
            A.append(lab[i])
        else:
            A.append(0)
    B.append(A)

D = np.array(B)
A = np.array(X_train)
B = np.transpose(A)
m, n = A.shape





def psvm_nl(P, Q, nu):
    m, n = P.shape
    e = np.ones(m, dtype=int)
    E = []
    for i in e:
        E1 = []
        E1.append(i)
        E.append(E1)
    e = np.array(E)
    S = np.append(P, -e, axis=1)

    H = np.matmul(Q, S)
    a = (np.identity(m) / nu + np.matmul(H, np.transpose(H)))
    v = np.matmul(np.linalg.inv(a), e)

    w = np.matmul(np.matmul(Q, np.transpose(P)), np.matmul(Q, v))
    b = np.matmul(-np.transpose(e), np.matmul(Q, v))

    return [v, w, b]


e = np.ones(m, dtype=int)
E = []
for i in e:
    E1 = []
    E1.append(i)
    E.append(E1)
e = np.array(E)

k = 5
n = len(X_train.axes[1])
m1 = len(X_train.axes[0])
fold = []
fold1 = []
for i in range(k):
    l = int(m1 / k)
    m = []

    m.append(X_train.iloc[i * l:(i + 1) * l, 0:n])
    fold1.append(y_train.iloc[i * l:(i + 1) * l])
    fold.append(m)

f = np.array(fold)
f2 = np.array(fold1)

from sklearn.metrics import accuracy_score

d = [1, 2, 3, 4, 5]
nu = [1, 10, 100, 1000, 10000, 100000, 1000000]
k1 = np.arange(k)
for l in nu:
    for c in d:

        Accuracy = 0
        for i in range(k):

            m = []
            m.append(i)

            trai = []
            tes = []
            for j in k1:

                if j not in m:
                    p = f[j, 0]

                    for r in range(len(p)):
                        p2 = p[r]
                        li = []
                        for r2 in range(len(p2)):
                            li.append(p2[r2])
                        trai.append(li)



                else:
                    p1 = f[j, 0]
                    y1 = f2[j]
                    for r1 in range(len(p1)):
                        p3 = p1[r1]
                        te = []
                        for r3 in range(len(p3)):
                            te.append(p3[r3])
                        tes.append(te)

            tee = []
            for i in y1:
                te = []
                te.append(i)
                tee.append(te)
            test = np.array(tes)

            y_testt = np.array(tee)
            train = np.array(trai)

            K = sklearn.metrics.pairwise.polynomial_kernel(train, train, degree=c, coef0=1)

            B = []

            for i in range(len(train)):
                A11 = []
                for j in range(len(train)):
                    if i == j:
                        A11.append(lab[i])
                    else:
                        A11.append(0)
                B.append(A11)

            D_ = np.array(B)
            t3 = psvm_nl(K, D_, i)
            y_cal = []
            e = np.ones(len(train), dtype=int)
            E = []
            for i in e:
                E1 = []
                E1.append(i)
                E.append(E1)
                e = np.array(E)

            for i in test:
                m = []
                m.append(i)
                m2 = np.array(m)

                t1 = np.matmul(sklearn.metrics.pairwise.polynomial_kernel(m2, train, degree=c, coef0=1),
                               np.transpose(K)) + np.transpose(e)
                y_cal.append(np.matmul(t1, np.matmul(D_, t3[0])))

            y_ob = []

            for i in y_cal:
                if i > 0:

                    y_ob.append(1)
                elif i <= 0:
                    y_ob.append(0)
            y_pred = np.array(y_ob)
            Accuracy = Accuracy + accuracy_score(y_testt, y_pred)

        print(f'({l},{c})= {Accuracy / k}')
    
d = 2
mu=10
e = np.ones(len(A), dtype=int)
E = []
for i in e:
    E1 = []
    E1.append(i)
    E.append(E1)
    e = np.array(E)
K= sklearn.metrics.pairwise.polynomial_kernel(A,A,degree=d,coef0=1)
t3=psvm_nl(K,D,mu)
t=np.array(X_test)
y_cal=[]
for i in t:
    
    m=[]
    for j in i:
        m.append(j)
    m2=[]
    m2.append(m)
    m1=np.array(m2)
    
    t1=np.matmul(sklearn.metrics.pairwise.polynomial_kernel(m1,A,degree=d,coef0=1),np.transpose(K))+ np.transpose(e)
    y_cal.append(np.matmul(t1,np.matmul(D,t3[0])))

y_ob=[]


for i in y_cal:
    if i>0:
        y_ob.append(1)
    elif i<=0:
        y_ob.append(0)
y_pred=np.array(y_ob)
print('Report of Non-Linear PSVM with Polynomial kernel is  ')
print(classification_report(y_test,y_pred))