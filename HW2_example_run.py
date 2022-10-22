import numpy as np
import pandas as pd
import sys
import time
import pickle
from Ensemble_Learning.Boosting import Boost
from Linear_Regression.Linear_Regression import LinearRegression, Analytical_w, Eval
import matplotlib.pyplot as plt
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
    bank_data_Train_o = pd.read_csv('./datasets/bank/train.csv')
    bank_data_Test_o = pd.read_csv('./datasets/bank/test.csv')
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


bagging = False
adaboost = False
randomforest = False
linearregression = True
method = 'BGD'
if bagging:
    # BAGGING
    method = 0 # 0:gini index, 1: majority error, 2:entropy
    depth = 16 # maximum depth
    T = 5 # number of Trees
    Y = df[labels] # ground truth
    n = 5000 # number of values to sample
       
     
    randomforest = False # set True for random forest 
    num_Attributes=None # number of attributes to randomly select
    
    boost = Boost(df, T, labels, Attribute_vals_df, method, Y, n, depth)
    boost.Bagging()
    data = boost.data

if adaboost:
    # ADABOOST
    method = 0 # 0:gini index, 1: majority error, 2:entropy
    depth = 1 # maximu
    T = 10 # number of Trees
    Y = df[labels] # ground truth
    n = 5000 # number of values to sample
    
    boost = Boost(df, T, labels, Attribute_vals_df, method, Y, n, depth)
    boost.AdaBoost()
    data = boost.data
    
if randomforest:
    # RANDOMFOREST
    method = 0 # 0:gini index, 1: majority error, 2:entropy
    depth = 16 # maximum depth
    T = 5 # number of Trees
    Y = df[labels] # ground truth
    n = 5000 # number of values to sample
       
     
    randomforest = True # set True for random forest 
    num_Attributes=3 # number of attributes to randomly select
    
    boost = Boost(df, T, labels, Attribute_vals_df, method, Y, n, depth)
    boost.Bagging(randomforest = randomforest, num_Attributes=num_Attributes)
    data = boost.data
    
if linearregression:
    data = pd.read_csv('HW2/LMS/data/slump_test.data')
    train=data.sample(frac=53/103,random_state=42)
    test=data.drop(train.index)

    attributes = data.columns[1:8]
    label = data.columns[-2]

    X = train[attributes]
    Y = train[label]

    X_test = test[attributes]
    Y_test = test[label]

    LR = LinearRegression(train, attributes, label)
    W = LR.LMS(method=method)
    J = LR.J(W.T, X, Y)
    J_test = LR.J(W.T, X_test, Y_test)
    J_True = LR.J(Analytical_w(X.T, Y).T, X, Y)

    #%%
    if method == 'BGD':
        plt.figure()
        plt.title('Batch Gradient Descent')
        np.log(J).plot(label='Train')
        np.log(J_test).plot(label='Test')
        plt.hlines(np.log(J_True), 0, len(J), colors=['k'], label='Analytical')
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('Log of Loss')
        plt.show()
        #plt.savefig('../images/BGD.png', dpi=300)
        # r = 1.1920928955078126e-08
        # W = array([ 0.        , -0.02500698, -0.11633336, -0.01940216,  0.47193326, -0.0100137 , -0.0236101 , -0.00645138])
        # 5e-6 tolorance
    #%%
    if method == 'SGD':
        plt.figure()
        plt.title('Stochastic Gradient Descent')
        np.log(J).plot(label='Train')
        np.log(J_test).plot(label='Test')
        plt.hlines(np.log(J_True), 0, len(J), colors=['k'], label='Analytical')
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('Log of Loss')
        plt.show()
        plt.savefig('../images/SGD.png', dpi=300)
        # r = 10e-09
        # w = array([ 0.        ,  0.00155264, -0.07931301, -0.00792159,  0.25177673, 0.038896  , -0.02083834,  0.03501546])
        # test_cost: 4438
        # train_cost: 4788
        # true_cost_train:4511
        #true_cost_test:1982



    