# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:59:38 2018

@author: Administrator
"""

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston


def ftest(X,y):
    # X: inpute variables
    # y: target
    X=np.c_[np.ones(n),X]
    xtx=np.matmul(X.T, X)
    xtx_inv=np.linalg.inv(xtx)
    beta=np.matmul(np.matmul(xtx_inv, X.T), y)
    y_pred=np.matmul(X,beta)
    SSE=sum((y-y_pred)**2)
    SSR=sum((y_pred-y.mean())**2)
    n=len(X)
    p=len(data.feature_names)
    MSE=SSE/(n-p-1)
    MSR=SSR/p
    f=MSR/MSE
    p_value=1-stats.f.cdf(f,p,n-p-1)
    alpha=0.1
    if p_value< alpha:
        ans=print("reject")
    else: ans=print("accept")   
    return ans

def ttest(X,y):
    # X: inpute variables
    # y: target
    n=len(X)
    p=len(data.feature_names)
    X=np.c_[np.ones(n),X]
    xtx=np.matmul(X.T, X)
    xtx_inv=np.linalg.inv(xtx)
    beta=np.matmul(np.matmul(xtx_inv, X.T), y)
    y_pred=np.matmul(X,beta)
    SSE=sum((y-y_pred)**2)
    MSE=SSE/(n-p-1)
    t=[]
    for i in range(p+1):
        t.append(beta[i]/np.sqrt(MSE*xtx_inv[i,i]))
    pvalue=[]
    for i in range(p+1):
        pvalue.append((1-stats.t.cdf(np.abs(t[i]),n-p-1))*2)


## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)


