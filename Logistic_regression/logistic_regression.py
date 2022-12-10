import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    data = pd.read_csv('../Data_sets/bank-note/train.csv')
    data_test = pd.read_csv('../Data_sets/bank-note/test.csv')
    data = pd.DataFrame(data.to_numpy(), columns=['x1', 'x2', 'x3', 'x4', 'y'])
    data['y'] = data['y'] * 2 - 1
    data_test['y'] = data_test['y'] * 2 - 1
    x = data[['x1', 'x2', 'x3', 'x4']].to_numpy()
    y = data['y'].to_numpy().reshape(-1, 1)
    x_test = data_test[['x1', 'x2', 'x3', 'x4']].to_numpy()
    y_test = data_test['y'].to_numpy().reshape(-1, 1)
    return x, y, x_test, y_test

Xtr, ytr, Xte, yte = get_data()



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class MAP:
    def __init__(self, X, y, std):
        self.X = X
        #import pdb;pdb.set_trace()
        self.X = np.column_stack((X, np.ones([X.shape[0], 1])))
        self.y = y
        self.std = std
        self.w = np.zeros(X.shape[1]+1)
    def loss(self, x, y, w):
        return np.log(1 + np.exp(-y * np.inner(w, x))) + 1/(self.std**2) * np.inner(w, w)
    
    def grad_loss(self, x, y, w):
        return -sigmoid(-y * np.inner(w, x))*(y*x) + 2/self.std * w
    
    def pred(self, X):
        return np.sign(np.inner(self.w[0:-1], X) + self.w[-1])
    
    def train_error(self):
        return np.count_nonzero(np.sign(np.inner(self.w, self.X)).flatten() * self.y.flatten() != 1)/len(self.X)
    
class ML:
    def __init__(self, X, y, std=None):
        self.X = X
        #import pdb;pdb.set_trace()
        self.X = np.column_stack((X, np.ones([X.shape[0], 1])))
        self.y = y
        self.w = np.zeros(X.shape[1]+1)
    def loss(self, x, y, w):
        return np.log(1 + np.exp(-y * np.inner(w, x)))
    
    def grad_loss(self, x, y, w):
        return -sigmoid(-y * np.inner(w, x))*(y*x)
    
    def pred(self, X):
        return np.sign(np.inner(self.w[0:-1], X) + self.w[-1])
    
    def train_error(self):
        return np.count_nonzero(np.sign(np.inner(self.w, self.X)).flatten() * self.y.flatten() != 1)/len(self.X)
    
def shuffle(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]
class Gradient_Descent:
    def __init__(self):
        return
    @staticmethod
    def optimize(model, epochs, gamma):
        w = model.w
        error_hist = []
        
        for t in range(epochs):
            X = model.X
            Y = model.y
            p = np.random.permutation(X.shape[0])
            for x, y in zip(X[p], Y[p]):
                w = w - gamma(t) * model.grad_loss(x, y, w)
            model.w = w
            error_hist.append(model.train_error())
        model.w = w
        return w, error_hist

def gamma(t):
    gamma0 = 1e-2
    d = 1e-4
    return gamma0 / (1 + gamma0/d * t)
V = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
epochs = 100
accuracyte = []
accuracytr = []
for v in V:
    std = v**2
    lr = ML(Xtr, ytr, std)
    w, error = Gradient_Descent.optimize(lr, epochs, gamma)
    hte = lr.pred(Xte)
    htr = lr.pred(Xtr)
    plt.plot(range(epochs), error)
    accuracyte.append(np.count_nonzero(hte*yte.flatten() != 1)/len(yte))
    accuracytr.append(np.count_nonzero(htr*ytr.flatten() != 1)/len(ytr))

