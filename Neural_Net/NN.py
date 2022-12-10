import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

class ANN:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        self.activation_fun = sigmoid
        self.d_activation_fun = dsigmoid
        
        self.layers = None
        self.Z = None
        return
    
    def forwardprop(self, x):
        Z = [x]
        for layer in self.layers:
            layer.z = Z[-1]
            Z.append(layer.forward_prop())
        self.Z = Z
        return Z
    
    def backprop(self):
        dw = []
        next_layer = None
        for i, layer in enumerate(self.layers[::-1]):
            dw.append(layer.back_prop(next_layer))
            next_layer = layer
            print(dw[-1])
        return
    
    def loss_fun(self, y, yh):
        return 0.5 * (y - yh)**2
    

    
class hidden_layer:
    def __init__(self, size, activation_fun, dactivation_fun):
        self.w = np.ones([size, size])
        self.z = np.zeros(size)
        self.activation_fun = activation_fun
        self.dactivation_fun = dactivation_fun
        return 
    
    def forward_prop(self):
        return self.activation_fun(np.inner(self.z, self.w.T))
    
    def back_prop(self, next_layer):
        self.dzdw = np.outer(self.dactivation_fun(np.inner(self.z, self.w.T)), self.z)# * next_layer.dydz
        self.dydz = self.dactivation_fun(np.inner(self.z, self.w.T)).reshape(-1,1) * self.w.T * next_layer.dydz
        return next_layer.dydz * self.dzdw
        
class output_layer:
    def __init__(self, size):
        self.w = np.ones(size)
        self.z = np.zeros(size)
        self.loss_fun = None
        return 
    
    def forward_prop(self):
        return np.inner(self.z, self.w.T)
    
    def back_prop(self, next_layer):
        self.dydw = self.z
        self.dydz = np.array([self.w,self.w,self.w]).T
        return self.z
        
    
    
    
        
    
if __name__ == '__main__':
    X = np.array([1, 1, 1])
    y = np.array([1])
    
    size = 3
    activation_fun = sigmoid
    dactivation_fun = dsigmoid
    layers = [hidden_layer(size, activation_fun, dactivation_fun), hidden_layer(size, activation_fun, dactivation_fun), output_layer(size)]
    
    layers[0].w =  np.array([[np.inf, -1, 1],
                             [0, -2, 2],
                             [0, -3, 3]])
    layers[1].w =  np.array([[np.inf, -1, 1],
                             [0, -2, 2],
                             [0, -3, 3]])
    layers[2].w = np.array([-1, 2, -1.5])
    
    NN = ANN(X, y)
    NN.layers = layers
    a = NN.forwardprop(X)
    NN.backprop()
    