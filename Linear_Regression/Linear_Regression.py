import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, data, attributes, label, r=1):
        self.df = data
        self.df['b'] = np.ones(len(self.df))
        self.data = self.df
        self.method = None
        self.optimizer = None
        self.attributes = attributes
        self.label = label
        self.W = np.zeros(len(attributes) + 1).T
        self.y = data[label]
        self.X = data[['b']+list(attributes)]
        self.r = 10e-09
        self.max_iter = int(10000)
        
    def LMS(self, method='BGD'):
        if method == 'BGD':
            w = self.Gradient_Descent(self.J, self.W, self.X)
        if method == 'SGD':
            w = self.SGD(self.J, self.W, self.X)
        return w
    
    
    def Gradient(self, W, X, Y):
        gradF = np.zeros(len(W))
        gradF = -(Y - W @ X.T) @ X
        return gradF
    
    def Gradient_Descent(self, Loss_fun, W, X):
        W = [W]
        loss = [Loss_fun(W[0], X.copy().drop('b', axis=1), self.y)]
        for i in range(self.max_iter):
            grad_loss = self.Gradient(W[i], X, self.y)
            W.append(W[i] - self.r*grad_loss)
            loss.append(Loss_fun(W[i], X.copy().drop('b', axis=1), self.y))
            
            if np.linalg.norm(W[-1]-W[-2]) < 5e-6:
                return np.array(W)
            if loss[i+1] > loss[i]:
                W[-2] = W[-3]
                W[-1] = W[-2]
                self.r *= 0.5
                print(self.r)
            
        return np.array(W)
    

    def SGD(self, Loss_fun, W, X):
        W = [W]
        for i in range(self.max_iter):
            for t in range(len(self.Y)):
                W.append(W[-1] + self.r*(self.Y.iloc[t] - W[-1] @ X.iloc[t])*X.iloc[t])
                # if np.linalg.norm(W[-1]-W[-2]) < 1e-6:
                #     return np.array(W)
        return np.array(W)

    
    def J(self, w, X, y):
        return self.Loss(self.Cost(self.Evaluate(w, X), y))
    
    def Cost(self, h, y):
        return abs((h.T - y.T).T)
    
    def Loss(self, cost):
        return np.sum(cost**2)/2
    
    def Evaluate(self, w, X):
        return w[0] + X @ w[1::]
    
def Eval(w, X):
    return w[0] + X @ w[1::]

def Analytical_w(X, Y):
    w = np.zeros(X.shape[0]+1)
    x = np.ones([X.shape[0]+1, X.shape[1]])
    x[1::] = X
    X = x
    w[0::] = np.linalg.inv(X @ X.T) @ (X @ Y)
    return w
    