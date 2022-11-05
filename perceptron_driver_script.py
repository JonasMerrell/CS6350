from Perceptron.Perceptron import standard, averaged, voted
import pandas as pd
import numpy as np


def get_data():
    data = pd.read_csv('./Data_sets/bank-note/train.csv')
    data_test = pd.read_csv('./Data_sets/bank-note/test.csv')
    data = pd.DataFrame(data.to_numpy(), columns=['x1', 'x2', 'x3', 'x4', 'y'])
    data['y'] = data['y'] * 2 - 1
    data_test['y'] = data_test['y'] * 2 - 1
    x = data[['x1', 'x2', 'x3', 'x4']]
    y = data['y']
    x_test = data_test[['x1', 'x2', 'x3', 'x4']]
    y_test = data_test['y']
    return x, y, x_test, y_test


x, y, x_test, y_test = get_data()

method = input('To choose standard method, input 1\nTo choose voted method, input 2\nTo choose averaged method, input 3\n>>> ')
if method in [1, '1', 'one', 'standard']:
    P = standard(x, y, r=1)
if method in [2, '2', 'two', 'voted']:
    P = voted(x, y, r=1)
if method in [3, '3', 'three', 'averaged']:
    P = averaged(x, y, r=1)
P.optimize(T=10)
w = P.w
w_array = P.w_array

y_ = P.predict(x_test)

error = np.sum(y_ * y_test != 1)/len(y_test) 
num_correct = np.sum(y_ == y_test)
if method in [1, '1', 'one']:
    print(f'The optimal weight vector is: {w[0:-1]}')
    print(f'The bias term is: {w[-1]}')
    
print(f'The test error is: {error}')
print(f'The number of correctly classified examples is: {num_correct}')

