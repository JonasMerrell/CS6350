import pandas as pd
import numpy as np


class perceptron:
    def __init__(self, X, y, r):
        X = X.sample(len(X))
        y = y.loc[X.index]
        self.X = X
        self.X['b'] = np.ones(X.shape[0]) 
        self.y = np.array(y)
        self.r = r
        self.w = np.zeros(X.shape[1])
        self.w_array = None
        self.C = None

class standard(perceptron):
    def __init__(self, X, y, r):
        perceptron.__init__(self, X, y, r)
        
    def optimize(self, T):
        w = [self.w]
        for t in range(T):
            w = [self.w]
            for i, x in enumerate(np.array(self.X)):
                if self.y[i] * x @ self.w <= 0:
                    self.w = self.w + self.r * (self.y[i] * x)
                    w.append(self.w)
        self.w_array = w

    def predict(self, x):
        x_ = x.copy()
        x_['b'] = np.ones(len(x))
        return np.sign(x_ @ self.w)
    
class voted(perceptron):
    def __init__(self, X, y, r):
        perceptron.__init__(self, X, y, r)
    
    def optimize(self, T):
        m = 0
        w = [self.w]
        C = [0]
        for t in range(T):
            for i, x in enumerate(np.array(self.X)):
                if self.y[i] * x @ self.w <= 0:
                    self.w = self.w + self.r * (self.y[i] * x)
                    w.append(self.w)
                    m += 1
                    C.append(1)
                else:
                    C[m] += 1
        self.C = np.array(C)
        self.w_array = np.array(w)
        
    def predict(self, x):
        x_ = x.copy()
        x_['b'] = np.ones(len(x))
        return np.sign(np.sum(np.sign(x_ @ self.w_array.T) * self.C, axis=1))
    
class averaged(voted):
    def __init__(self, X, y, r):
        voted.__init__(self, X, y, r)
        
    def predict(self, x):
        x_ = x.copy()
        x_['b'] = np.ones(len(x))
        return np.sign(np.sum(x_ @ self.w_array.T * self.C, axis=1))
        

if __name__ == '__main__':
    def get_data():
        data = pd.read_csv('../Data_sets/bank-note/train.csv')
        data_test = pd.read_csv('../Data_sets/bank-note/test.csv')
        data = pd.DataFrame(data.to_numpy(), columns=['x1', 'x2', 'x3', 'x4', 'y'])
        data['y'] = data['y'] * 2 - 1
        data_test['y'] = data_test['y'] * 2 - 1
        x = data[['x1', 'x2', 'x3', 'x4']]
        y = data['y']
        x_test = data_test[['x1', 'x2', 'x3', 'x4']]
        y_test = data_test['y']
        return x, y, x_test, y_test


    x, y, x_test, y_test = get_data()
    W = []
    C = []
    for method in [2,3]:
        if method in [1, '1', 'one']:
            P = standard(x, y, r=1)
        if method in [2, '2', 'two']:
            P = voted(x, y, r=1)
        if method in [3, '3', 'three']:
            P = averaged(x, y, r=1)
        P.optimize(T=10)
        w = P.w
        w_array = P.w_array
        W.append(w_array)
        C.append(P.C)
        y_ = P.predict(x_test)
    
        error = np.sum(y_ * y_test != 1)/len(y_test) 
        num_correct = np.sum(y_ == y_test)

    wc = np.column_stack((W[0], C[0]))

