import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def logistic():
    X = np.array([[0.5, -1, 0.3, 1],
                  [-1, -2, -2, 1],
                  [1.5, 0.2, -2.5, 1]])
    y = np.array([1, -1, 1])
    
    w = np.array([0., 0., 0., 0.])
    
    gamma = np.array([0.01, 0.005, 0.0025])
    
    ws = []
    grad = []
    for i in range(3):
        ws.append(w-gamma[i]*sigmoid(-y[i]*np.inner(w, X[i])) * (-y[i]* X[i]) + 2*w)
        w -= gamma[i]*sigmoid(-y[i]*np.inner(w, X[i])) * (-y[i]* X[i]) + 2*w
        print(w)
        
        grad.append(gamma[i]*sigmoid(-y[i]*np.inner(w, X[i])) * (-y[i]* X[i]) + 2*w)
    
        
    ws = np.array(ws)
    grad = np.array(grad)



y = 1
w = [None, None, None]
w[2] =  np.array([[np.inf, -1, 1],
                         [0, -2, 2],
                         [0, -3, 3]])
w[1] =  np.array([[np.inf, -1, 1],
                         [0, -2, 2],
                         [0, -3, 3]])
w[0] = np.array([-1, 2, -1.5])

Z = np.array([[1, 0.018, 0.982],
              [1, 0.00247, 0.998],
              [1, 1, 1]])
              
h = -2.437

dLdy = h - y
dydw3 = Z[0]
dydz2 = w[0]

dz2dw2 = np.outer(dsigmoid(np.inner(Z[1], w[1].T)), Z[1])
dz2dz1 = dsigmoid(np.inner(Z[1], w[1].T)).reshape(-1,1) * w[1].T

dz1dw1 = np.outer(dsigmoid(np.inner(Z[2], w[2].T)), Z[2])

dLdw3 = np.array(dydw3)
dLdw2 = np.array((dydz2 * dz2dw2.T).T)
dLdw1 = np.array((dydz2 * dz2dz1.T).T * dz1dw1)

dLdw = np.row_stack((dLdw3, dLdw2[1::], dLdw1[1::]))
