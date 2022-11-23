import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.spatial.distance

class SVM:
    def __init__(self, X, y, C):
        X = X.copy()
        self.X_only = np.array(X).T.copy()
        X['b'] = np.ones(X.shape[0])
        self.X = np.array(X).T
        self.y = np.array(y).reshape(1,-1)
        self.w = np.zeros(X.shape[1]).reshape(-1,1)
        self.w
        self.w_array = None
        self.C = C
        self.N = len(y)
        self.errors = []
        
    def shuffle(self, X, y):
        X = X.T
        y = y.T
        assert len(X) == len(y)
        p = np.random.permutation(len(X))
        return X[p].T, y[p].T

    def sub_gradient_descent(self, T=100, gamma=None):
        w = self.w
        for t in range(T):
            #gamma_t = gamma_0 / (1 + gamma_0 * t / a)
            if gamma is None:
                gamma_t = 1 / (1 + t)
            else:
                gamma_t = gamma(t)
                
            X, Y = self.shuffle(self.X, self.y)
            #X, Y = self.X, self.y
            for x, y in zip(X.T, Y.T):
                y = float(y)
                x = x.reshape(-1, 1)
                #import pdb;pdb.set_trace()
                if w.T @ x * y <= 1:
                    
                    w = w - gamma_t * np.append(w[0:-1],0).reshape(-1,1) + gamma_t * self.C * self.N * y * x
                    print(gamma_t * np.append(w[0:-1],0).reshape(-1,1) + gamma_t * self.C * self.N * y * x)
                else:
                    w[0:-1] = (1-gamma_t) * w[0:-1]
                    
            self.errors.append(self.error(Y.flatten(), self.evaluate(w, X).flatten()))
        return w
    
    @staticmethod
    def evaluate(w, X, b=None):
        #import pdb;pdb.set_trace()
        w = w.flatten().reshape(-1, 1)
        if b is None:
            return np.sign(w.T @ X)
        else:
            return np.sign(w.T @ X + b)
        
    @staticmethod
    def error(y, y_):
        return np.count_nonzero(y*y_ == -1)/len(y)     
    
class dual(SVM):
    def __init__(self, X, y, C):
        SVM.__init__(self, X, y, C)
        self.maxiter = None
    
    def dual_SVM(self, a, args=[None]):
        y = self.y.flatten()
        a = a.flatten()
        X = self.X_only.T
        #return  0.5 * (np.sum(a.T @ ((X @ X.T) *(y @ y.T)) @ a)) - np.sum(a)
        if args[0] is None:
            return 0.5 * np.sum(np.outer(a,a) * np.outer(y,y) * np.dot(X, X.T)) - np.sum(a)
        else:
            return 0.5 * np.sum(np.outer(a,a) * np.outer(y,y) * args[0](X, args[1])) - np.sum(a)
    
    def dual_SVM2(self, a):
        s = 0
        y = self.y.flatten()
        X = self.X_only.T
        for i in range(len(a)):
            for j in range(len(a)):
                s += y[i] * y[j] * a[i] * a[j] * X[i] @ X[j].T 
        s *= 0.5
        s -= np.sum(a)
        return s
    
    def kernal(self, X, gamma, z=None):
        if z is None:
            z = X
        return np.exp(-scipy.spatial.distance.cdist(X, z,'sqeuclidean')/gamma) 
    
    def optimize(self, kernal=None, gamma=None):
        y = self.y.flatten()
        a0 = np.zeros(len(y))
        constraint = lambda x: np.dot(x.flatten(), y.flatten())
        a = scipy.optimize.minimize(self.dual_SVM, a0, method='SLSQP', bounds=[[0, self.C]]*len(a0), 
                                    options={'maxiter':self.maxiter},
                                    constraints=[{"type":"eq", "fun":constraint}],
                                    args=[kernal, gamma])
        return a.x   
    

    def weights(self, a):
        w = 0
        for i in range(len(a)):
            #import pdb;pdb.set_trace()
            w += a[i] * self.y.flatten()[i] * self.X_only.T[i]
        return np.array(w)

    def weights_b(self, kernal=None, gamma=None):
        a = self.optimize(kernal, gamma)
        w = self.weights(a)
        y = self.y.flatten()
        b = np.mean(y - np.dot(w,self.X_only))
        return w, b
                            
    def evaluate2(self, a, X, gamma, kernal):
        #import pdb;pdb.set_trace()
        b = np.mean(self.y - np.sum(a.flatten() * self.y.flatten() * kernal(self.X_only.T, gamma), axis=0))
        #import pdb;pdb.set_trace()
        return np.sign(np.sum((a.flatten() * self.y.flatten()).reshape(-1,1) * kernal(self.X_only.T, gamma, z=X), axis=0)+b)                  
     
#%%
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




    #%%
    import numpy as np
    X = np.array([[0.5, -1, 0.3,1],
                  [-1, -2, -2,1],
                  [1.5, 0.2, -2.5,1]])

    y = np.array([1,-1,1])
    r = [0.01, 0.005, 0.0025]
    w = np.zeros(4)
    
    for i in range(3):
        
        w = w - r[i] * np.append(w[0:-1],0) + r[i] * 1 * 3 * y[i] * X[i]
        print(f'after step {i}:')
        print(f'w = {w}')
        
    y_ = np.sign(w @ X.T)
    
    #%%

    X, y, X_test, y_test = get_data()
    X_test['b'] = np.ones(len(X_test))
    Xt = X.copy()
    Xt['b'] = np.ones(len(Xt))
    C = np.array([100/873, 500/873, 700/873])
    gamma = np.array([0.1, 0.5, 1, 5, 100])
    #%%
    def Gamma(t):
        g0 = 0.01
        return g0 / (1 + t)
    
    # def Gamma(t):
    #     g0 = 1
    #     a = 0.005
    #     return g0/(1 + g0/a * t)
    
    plt.figure()
    error_train = []
    error_test = []
    ws = []
    for c in C:
        svm = SVM(X, y, c)
        w = svm.sub_gradient_descent(T=100, gamma=Gamma)
        ws.append(w)
        plt.plot(range(len(svm.errors)), svm.errors, label=str(c))
        y_ = np.array(SVM.evaluate(w, X_test.T)).flatten()
        y_t = np.array(SVM.evaluate(w, Xt.T)).flatten()
        error_test.append(SVM.error(y_test, y_))
        error_train.append(SVM.error(y, y_t))
    plt.legend()
    ws = np.array(ws).reshape(-1,5)
    

    #%%
    error_train = []
    error_test = []
    ws = []
    for c in C:
        dualsvm = dual(X, y, c)
        dualsvm.maxiter = 10
        w, b = dualsvm.weights_b()
        y_ = dualsvm.evaluate(w, X_test[['x1','x2','x3','x4']].T, b)
        yt = dualsvm.evaluate(w, X[['x1','x2','x3','x4']].T, b)
        error_test.append(SVM.error(y_test, y_))
        error_train.append(SVM.error(y, yt))
        ws.append(w)
    #%%
    if 1==1:
        alphas = []
        for c in C:
            for g in [gamma[4]]:
                print(c)
                print(g)
                dualsvm = dual(X, y, c)
                dualsvm.maxiter = 10
                alphas.append(dualsvm.optimize(dualsvm.kernal, g))

    
    #%%
    dualsvm = dual(X, y, c)
    y__ = dualsvm.evaluate2(alphas[0], X, y, gamma[0], dualsvm.kernal)
    error2 = SVM.error(y, y__)   
    
    #%% 
    alpha = np.load('alpha.npy')
    alpha = np.row_stack((alpha, np.load('alphas4.npy')))
    issup = alpha.copy()
    issup = ~np.isclose(issup, 0)
    error_train = np.ones([len(C), len(gamma)])
    error_test = np.ones([len(C), len(gamma)])
    issupc = []
    s = 0
    for i, c in enumerate(C):
        for j, g in enumerate(gamma):
            dualsvm = dual(X, y, c)
            yt = dualsvm.evaluate2(alpha[s], X, g, dualsvm.kernal)
            y_ = dualsvm.evaluate2(alpha[s], X_test[['x1','x2','x3','x4']], g, dualsvm.kernal)
            error_train[i,j] = SVM.error(y, yt)
            error_test[i,j] = SVM.error(y_test, y_)
            if c == C[2]:
                issupc.append(issup[s])
            
            s += 1
    issupc = np.array(issupc)
    numsupc = np.count_nonzero(issupc, axis=1)
    
    overlap = []
    for i in range(len(gamma)-1):
        overlap.append((issupc[i] == issupc[i+1]) * issupc[i])
    overlap = np.array(overlap)
    
    numoverlap = np.count_nonzero(overlap, axis=1)
        