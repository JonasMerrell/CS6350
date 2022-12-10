import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd

#from misc import *

import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm, trange

device = torch.device('cuda:0')

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


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index,:], self.y[index,:]

    def __len__(self,):
        return self.X.shape[0]
    
    
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def train(config, lr, reg, epochs, model, early_stopping=None):
    
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print('r', 'Total params: {:.3f}K'.format(num_params/(1e3)))
    
    train_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    
    hist_rmse_tr = []
    hist_rmse_te = []
    hist_pred_te = []
    hist_pred_tr = []
    
    for ie in tqdm(range(epochs+1)):
        
        Xtr, ytr = next(iter(train_loader))
        Xte, yte = next(iter(test_loader))
        
        Xtr, ytr, Xte, yte = \
            Xtr.float().to(device), ytr.float().to(device), Xte.float().to(device), yte.float().to(device)
        
        pred = model(Xtr)
        loss = torch.mean(torch.square(pred-ytr))
        
    
        # step 1: clear the grads
        optimizer.zero_grad()
        # step 2: backward the computational graph
        loss.backward()
        # step 3: take the gradient step
        optimizer.step()
        
        if ie%500 == 0:
            print('r', 'Epoch #{}\t: '.format(ie), end='')
            with torch.no_grad():
                rmse_tr = torch.sqrt(torch.mean(torch.square(ytr-pred)))
                rmse_te = torch.sqrt(torch.mean(torch.square(yte-model(Xte))))
                print('train_rmse={:.5f}, test_rmse={:.5f}'.format(rmse_tr.item(), rmse_te.item()))
                
                hist_rmse_tr.append(rmse_tr.item())
                hist_rmse_te.append(rmse_te.item())
                hist_pred_te.append(np.sign(model(Xte).data.cpu().numpy()))
                hist_pred_tr.append(np.sign(model(Xtr).data.cpu().numpy()))
                
                
                losste = torch.mean(torch.square(model(Xte)-yte))
                if early_stopping is not None:
                    early_stopping(loss, losste)
                    if early_stopping.early_stop:
                      print("We are at epoch:", ie)
                      break
    
    hist_rmse_tr = np.array(hist_rmse_tr)
    hist_rmse_te = np.array(hist_rmse_te)
    hist_pred_te = np.array(hist_pred_te)
    
    
    
    fig = plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')
    hist = np.arange(hist_rmse_tr.size)*500
    
    plt.plot(hist, hist_rmse_tr, color='b', label='train')
    plt.plot(hist, hist_rmse_te, color='r', label='test')
    plt.xlabel('num of epochs')
    plt.ylabel('RMSE')
    plt.legend()

    return model

class Net(nn.Module):
    def __init__(self, config, act=nn.Tanh(), weight_init=None):
        
        super(Net, self).__init__()
 
        layers_list = []
        
        
        for l in range(len(config)-2):
            in_dim = config[l]
            out_dim = config[l+1]
            
            
            layer = nn.Linear(in_features=in_dim, out_features=out_dim)
            if weight_init is not None:
                weight_init(layer.weight)
            layers_list.append(layer)
            layers_list.append(act)
            
        #
        
        # last layer
        layers_list.append(nn.Linear(in_features=config[-2], out_features=config[-1]))
        
        # containers: https://pytorch.org/docs/stable/nn.html#containers
        self.net = nn.ModuleList(layers_list)
        
    def forward(self, X):
        h = X
        for layer in self.net:
            h = layer(h)
        #
        return h
    
    
Xtr, ytr, Xte, yte = get_data()
#%%
dataset_train = SimpleDataset(Xtr, ytr)
dataset_test = SimpleDataset(Xte, yte)
epochs=5000
lr=10e-3
reg=1e-5
config = [4, 64, 64, 1]

early_stopping = EarlyStopping(tolerance=5, min_delta=10)

def He(layer):
    return torch.nn.init.kaiming_uniform_(tensor=layer, a=0, mode='fan_in', nonlinearity='relu')
def Xavier(layer):
    return torch.nn.init.xavier_normal_(tensor=layer, gain=1.0)

# model = Net(config, act=nn.Tanh(), weight_init=Xavier).to(device)
# train(config, lr, reg, epochs, model, early_stopping=early_stopping)
#%%
models=[]
for act, weight_init in zip([nn.ReLU(), nn.Tanh()], [He, Xavier]):
    for layers in [3, 5, 9]:
        for nodes in [5, 10, 25, 50, 100]:
            config = [4] + [nodes]*layers + [1]
            model = Net(config, act=act, weight_init=weight_init).to(device)
            models.append(train(config, lr, reg, epochs, model, early_stopping=early_stopping))
#%%
Xtr, ytr, Xte, yte = get_data()
Xte = torch.from_numpy(Xte).float().to(device)
Xtr = torch.from_numpy(Xtr).float().to(device)
def error(y, y_):
    return np.mean(abs((y-y_)/y))
for model in models:
    predtr = model(Xtr).to('cpu').detach().numpy()
    predte = model(Xte).to('cpu').detach().numpy()
    accuracytr = np.count_nonzero(np.sign(predtr).flatten() * ytr.flatten() == 1)/len(ytr)
    accuracyte = np.count_nonzero(np.sign(predte).flatten() * yte.flatten() == 1)/len(yte)
    errortr = error(predtr, ytr)
    errorte = error(predte, yte)

s = 0
tabletr = np.zeros([3,5,2])
tablete = np.zeros([3,5,2])
tabletra = np.zeros([3,5,2])
tabletea = np.zeros([3,5,2])
for l, (act, weight_init) in enumerate(zip([nn.ReLU(), nn.Tanh()], [He, Xavier])):
    for i, layers in enumerate([3, 5, 9]):
        for j, nodes in enumerate([5, 10, 25, 50, 100]):
            model = models[s]
            s += 1
            predtr = model(Xtr).to('cpu').detach().numpy()
            predte = model(Xte).to('cpu').detach().numpy()
            accuracytr = np.count_nonzero(np.sign(predtr).flatten() * ytr.flatten() == 1)/len(ytr)
            accuracyte = np.count_nonzero(np.sign(predte).flatten() * yte.flatten() == 1)/len(yte)
            errortr = error(predtr, ytr)
            errorte = error(predte, yte)
            tabletr[i,j,l] = errortr
            tablete[i,j,l] = errorte
            tabletra[i,j,l] = accuracytr
            tabletea[i,j,l] = accuracyte
            
    

