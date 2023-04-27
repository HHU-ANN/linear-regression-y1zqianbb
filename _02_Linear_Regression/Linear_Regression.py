# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y=read_data()
    alpha=0.25 #设置超参数alpha
    xTx=np.dot(x.T,x)
    rxTx=xTx+np.eye(x.shape[1])*alpha
    weight=np.dot(np.linalg.inv(rxTx),np.dot(x.T,y))
    return data @ weight
    


def lasso(data):
    X,y=read_data()
    lambdas=0.05
    max_iter=1000
    tol=1e-4
    w = np.zeros(X.shape[1])
    for it in range(max_iter):
        done = True
        for i in range(0, len(w)):
            weight = w[i]
            w[i] = down(X, y, w, i, lambdas)
            if (np.abs(weight - w[i]) > tol):
                done = False
        if (done):
            break
    return data @ w

def down(X, y, w, index, lambdas=0.05):
    aa = 0
    ab = 0
    for i in range(X.shape[0]):
        a = X[i][index]
        b = X[i][:].dot(w) - a * w[index] - y[i]
        aa = aa + a * a
        ab = ab + a * b
    return det(aa, ab, lambdas)

def det(aa, ab, lambdas=0.05):
    w = - (2 * ab + lambdas) / (2 * aa)
    if w < 0:
        w = - (2 * ab - lambdas) / (2 * aa)
        if w > 0:
            w = 0
    return w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

