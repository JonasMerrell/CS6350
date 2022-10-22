import numpy as np
import pandas as pd
import sys
import time
import pickle
sys.path.append('../Decision_Tree/')
from Decision_Tree import DecisionTree
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


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


labels = 'y'
bank_data_Train, bank_data_Test, Attribute_vals_df = get_input_data(labels)
df = bank_data_Train
method = 0
depth = 16
T = 500# number of steps
T_mpi = int(T/20)
Y = df[labels]
n = 5000

randomforest = False
test_data = bank_data_Test
mpi = True

for i in range(1):
    boost = Boost(df, T_mpi, labels, Attribute_vals_df, method, Y, n, depth)
    boost.Bagging_MPI(replacement=False, test_data=test_data)
    data , data_test = boost.Bagging_MPI(replacement=True, test_data=test_data)
    np.save(f'bags_{rank}_{i}_train.pkl', data)
    np.save(f'bags_{rank}_{i}_test.pkl', data_test)


