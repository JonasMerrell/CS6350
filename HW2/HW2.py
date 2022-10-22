import numpy as np
import pandas as pd
import sys
import time
import pickle
from Boosting import Boost
#%%

def Get_Most_Common(df, A):
     unique, counts = np.unique(df[A], return_counts=True)
     value = unique[np.argmax(counts)]
     return value
 
def Number_To_Binary(df, numeric_columns):

    for col in numeric_columns:
        median = np.median(df[col])
        large_locs = np.where(df[col] > median)[0]
        small_locs = np.where(df[col] <= median)[0]
        df[col].iloc[large_locs] = 1
        df[col].iloc[small_locs] = 0
    return df


def Pre_proc(df, numeric_columns, labels, fix_unknown = False):
    df_proc = df.copy()
    df_proc = Number_To_Binary(df_proc, numeric_columns)
    if fix_unknown:
        for A in df_proc.columns:
            locs_unknown = np.where(df_proc[A] == 'unknown')[0]
            if len(locs_unknown) == 0:
                continue
            else:
                df_proc[A].iloc[locs_unknown] = Get_Most_Common(df_proc, A)
    df_proc[labels].loc[df_proc[labels] == 'no'] = -1
    df_proc[labels].loc[df_proc[labels] == 'yes'] = 1
    df_proc[labels] = pd.to_numeric(df_proc[labels])
    return df_proc

def get_input_data(labels):
    bank_data_Train_o = pd.read_csv('../datasets/bank/train.csv')
    bank_data_Test_o = pd.read_csv('../datasets/bank/test.csv')
    numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    keys = ['age', 'job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous', 'poutcome', 'y']
    
    vals = ((0,1),
            ("admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"),
            ("married","divorced","single"),
            ("unknown","secondary","primary","tertiary"),
            ("yes","no"),
            (0,1),
            ("yes","no"),
            ("yes","no"),
            ("unknown","telephone","cellular"),
            (0,1),
            ("jan", "feb", "mar","apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
            (0,1),
            (0,1),
            (0,1),
            (0,1),
            ("unknown","other","failure","success"),
            ("yes","no"))
    
    bank_data_Train = Pre_proc(bank_data_Train_o, numeric_columns, labels, fix_unknown = False)
    bank_data_Test = Pre_proc(bank_data_Test_o, numeric_columns, labels, fix_unknown = False)
    Attribute_vals_df_bank = dict(zip(keys[0:-1], vals[0:-1]))
    return bank_data_Train, bank_data_Test, Attribute_vals_df_bank

labels = 'y'
bank_data_Train, bank_data_Test, Attribute_vals_df = get_input_data(labels)
df = bank_data_Train
test_data = bank_data_Test

if 1 ==  0:
    method = 0
    depth = 1
    T = 500# number of steps
    Y = df[labels]
    n = 5000
    
    
    
    randomforest = True
    
    mpi = False
    num_Attributes=1
    
    t0 = time.time()
    #boost.Bagging(test_data=test_data, mpi=mpi, randomforest=randomforest, num_Attributes=num_Attributes)
    boost = Boost(df, T, labels, Attribute_vals_df, method, Y, n, depth)
    boost.AdaBoost(replacement=False, test_data=test_data)
    t1 = time.time()
    total = t1-t0
    data = boost.data
    data_test = boost.data_test
    #%%
    Y_train = bank_data_Train[labels]
    Y_test = bank_data_Test[labels]
    error_local_test = np.zeros(T)
    error_local_train = np.zeros(T)
    
    error_test = np.zeros(T)
    error_train = np.zeros(T)
    
    accuracy_test = np.zeros(T)
    h_test = data_test['h']
    H_test = data_test['H']
    
    h_train = data['h']
    H_train = data['H']
    D = data['D']
    for t in range(T):
        error_local_test[t] = boost.Error(h_test[t]*Y_test, D[t])
        error_local_train[t] = boost.Error(h_train[t]*Y_train, D[t])
        
        error_test[t] = np.sum(H_test[t]*Y_test != 1)/5000
        error_train[t] = np.sum(H_train[t]*Y_train != 1)/5000

    import matplotlib.pyplot as plt
    plt.plot(range(T), error_local_test, 'b-', label='test')
    plt.plot(range(T), error_local_train, 'r-', label='train')
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('error')
    plt.title('single stump')
    
    plt.figure()
    plt.plot(range(T), error_test, 'b-', label='test')
    plt.plot(range(T), error_train, 'r-', label='train')
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('error')
    plt.title('all stumps')

#%%
import matplotlib.pyplot as plt
Y_train = bank_data_Train[labels]
Y_test = bank_data_Test[labels]
if 1 == 1:
    
    # BAGGING
    if 1 == 0:
        h = []
        h_test = []
        for cpu in range(20):
            h.append(np.load(f'bags_{cpu}_{0}_train.pkl.npy'))
            h_test.append(np.load(f'bags_{cpu}_{0}_test.pkl.npy'))
        H = np.array(h).reshape(500,5000)
        H_test = np.array(h_test).reshape(500,5000)
        accuracy = []
        accuracy_test = []
        for i in range(500):
            accuracy.append(np.sum(np.sign(np.mean(H[0:i+1,:], axis=0))*Y_train==1)/5000)
            accuracy_test.append(np.sum(np.sign(np.mean(H_test[0:i+1,:], axis=0))*Y_test==1)/5000)
    
    if 1 == 0:      
        plt.plot(range(500), accuracy, 'r-', label='Training Accuracy')
        plt.plot(range(500), accuracy_test, 'b-', label='Test Accuracy')
        plt.xlabel('T')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.figure()
        plt.plot(range(500), 1-np.array(accuracy), 'r-', label='Training Error')
        plt.plot(range(500), 1-np.array(accuracy_test), 'b-', label='Test Error')
        plt.xlabel('T')
        plt.ylabel('Error')
        plt.legend()
        
    
    
    # WHACK PROBLEM THAT TOOK WAY TOO MUCH CPU TIME
    if 1 == 1:
        if 1 ==1:
            H = []
            H_test = []
            for i in range(100):
                h = []
                h_test = []
                for cpu in range(20):
                    h.append(np.load(f'./npy_files/group_bags_{cpu}_{i}_train.pkl.npy'))
                    h_test.append(np.load(f'./npy_files/group_bags_{cpu}_{i}_test.pkl.npy'))
                h = np.array(h).reshape(500,5000)
                h_test = np.array(h_test).reshape(500,5000)
                H.append(np.sign(np.mean(h, axis = 0)))
                H_test.append(np.sign(np.mean(h_test, axis = 0)))
        H_test = np.array(H_test)
        # average = np.mean(H_test, axis=0)
        # bias = np.mean((average - Y_test)**2)
        # variance = np.sum((average - np.mean(average))**2)/4999
        # GSE = variance + bias
        
        # single_bias = np.mean((H_test[0] - Y_test)**2)
        # single_variance = np.sum((H_test[0] - np.mean(H_test[0]))**2) / 4999
        # single_GSE = single_variance + single_variance
        
        single_bias = np.mean((H_test[0]-Y_test)**2)
        single_variance = np.sum((H_test[0] - np.mean(H_test[0]))**2)/4999
        single_GSE = single_variance + single_bias

        bias = np.mean(np.mean((H_test - np.array(list(Y_test)*100).reshape(100, 5000))**2, axis=1))
        variance = np.mean(np.sum((H_test - np.array(list(np.mean(H_test, axis=1))*5000).reshape(5000, 100).T)**2, axis=1)/4999)
        GSE = variance + bias
    #RANDOM FOREST
    
    if 1 == 0:
        H_test = []
        H_train = []
        Error_test = []
        Error_train = []
        for i in [2,4,6]:
            for cpu in range(1):
                with open(f'pkl_files/RF_{cpu}_{i}_test.pkl', 'rb') as f:
                    h_test = pickle.load(f)['h']    
                with open(f'pkl_files/RF_{cpu}_{i}_train.pkl', 'rb') as f:
                    h_train = pickle.load(f)['h']
            
            error_test = np.zeros(500)
            error_train = np.zeros(500)
            for i in range(500):
                error_test[i] = np.sum(np.sign(np.mean(h_test[0:i+1], axis=0))*Y_test != 1)/5000
                error_train[i] = np.sum(np.sign(np.mean(h_train[0:i+1], axis=0))*Y_train != 1)/5000

            H_test.append(h_test)
            H_train.append(h_train)    
            Error_test.append(error_test)
            Error_train.append(error_train)        
            
        plt.plot(range(500), np.array(Error_test).T, '-', label=['2 Attributes test','4 Attributes test', '6 Attributes test'])
        plt.plot(range(500), np.array(Error_train).T, '-', label=['2 Attributes train','4 Attributes train', '6 Attributes train'])
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('error')
        
        
        #%%
        H_test = np.array(H_test).reshape(1500, 5000)
        single_bias = np.mean((H_test[0]-Y_test)**2)
        single_variance = np.sum((H_test[0] - np.mean(H_test[0]))**2)/4999
        single_GSE = single_variance + single_bias

        bias = np.mean(np.mean((H_test - np.array(list(Y_test)*1500).reshape(1500, 5000))**2, axis=1))
        variance = np.mean(np.sum((H_test - np.array(list(np.mean(H_test, axis=1))*5000).reshape(5000, 1500).T)**2, axis=1)/4999)
        GSE = variance + bias
        
                    
            
        
            

        

    