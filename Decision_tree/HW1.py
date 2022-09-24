import numpy as np
import pandas as pd
from Decision_Tree import DecisionTree


def Get_Most_Common(df, A):
     unique, counts = np.unique(df[A], return_counts=True)
     value = unique[np.argmax(counts)]
     return value

            

# #%% Problem 1.1
# S = np.array([[0,0,1,0,0],
#               [0,1,0,0,0],
#               [0,0,1,1,1],
#               [1,0,0,1,1],
#               [0,1,1,0,0],
#               [1,1,0,0,0],
#               [0,1,0,1,0]])
# A = np.array(['x1','x2','x3','x4','y'])
# data = pd.DataFrame(S, columns=A)
# Attributes = A[0:-1]
# Label = A[-1]
# vals_df = {'x1':[0,1],'x2':[0,1],'x3':[0,1],'x4':[0,1]}
# DT = DecisionTree(data, Attributes, Label, Attribute_vals_df=vals_df,method=2)
# DT.Train()
# import itertools
# lst = list(itertools.product([0, 1], repeat=4))
# df = pd.DataFrame(lst, columns=['x2','x4','x1','x3'])
# y = DT.Predict(df)

#%% Problem 1.2
#{Outlook: Missing, Temperature: Mild, Humidity: Normal, Wind: Weak, Play: Yes}
def fractional(S):
    s = S.copy()
    for i, val in enumerate(S[:,0]):
        if val != 'unknown':
            for j in range(len(S)):
                s = np.append(s, [S[i]], axis=0)
        else:
            locs_unknown = np.where(s[:,0] == 'unknown')[0]
            s  = np.delete(s, locs_unknown, axis=0)
            unique, counts = np.unique(S[:,0], return_counts=True)
            
            for j in range(len(unique)):
                for k in range(counts[j]):
                    a = S[i]
                    a[0] = unique[j]
                    s = np.append(s, [a], axis=0)
        
    return s[0:-1,:]
S = np.array([['S','H','H','W','-'],
              ['S','H','H','S','-'],
              ['O','H','H','W','+'],
              ['R','M','H','W','+'],
              ['R','C','N','W','+'],
              ['R','C','N','S','-'],
              ['O','C','N','S','+'],
              ['S','M','H','W','-'],
              ['S','C','N','W','+'],
              ['R','M','N','W','+'],
              ['S','M','N','S','+'],
              ['O','M','H','S','+'],
              ['O','H','N','W','+'],
              ['R','M','H','S','-'],
              ['unknown','M','N','W','+']])
A = np.array(['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play'])
S = fractional(S)
df = pd.DataFrame(S, columns=A)

if 1 == 0:
    for a in df.columns:
        locs_p = np.where(df['Play'] == '+')[0] 
        locs_unknown = np.where(df[a] == 'unknown')[0]
        if len(locs_unknown) == 0:
            continue
        else:
            df[a].iloc[locs_unknown] = Get_Most_Common(df.iloc[locs_p], a)

Label = A[-1]
Attributes = A[0:-1]

Attribute_vals_df = {'Outlook':['S','R','O'],'Temperature':['H','M','C'], 'Humidity':['H','N','L'], 'Wind':['S','W']}
DT = DecisionTree(df, Attributes, Label,Attribute_vals_df=Attribute_vals_df, method=2)
DT.Train()
y = DT.Predict(df)

# #%% Problem 2.2  
# car_data_Train = pd.read_csv('car/train.csv')
# car_data_Test = pd.read_csv('car/test.csv')
# Attribute_vals_df = {'buying':['vhigh', 'high', 'med', 'low'],'maint':['vhigh', 'high', 'med', 'low'],'doors':['2','3','4','5more'], 'persons':['2','4','more'],'lug_boot':['small','med','big'],'safety':['low','med','high']}
# acc = []
# acc_train = []
# for method in [0,1,2]:
#     acc_m = []
#     acc_train_m = []
#     for d in range(7):
#         print(d)
#         DT_car = DecisionTree(car_data_Train, car_data_Train.columns.to_list()[0:-1], car_data_Train.columns.to_list()[-1], Attribute_vals_df=Attribute_vals_df, depth=d, method=method)
#         DT_car.Train()
        
#         car_y = DT_car.Predict(car_data_Test)
    
#         car_accuracy = DT_car.Accuracy(car_data_Test['label'])
#         acc_m.append(car_accuracy)
    
#         car_y = DT_car.Predict(car_data_Train)
                         
#         car_accuracy_train = DT_car.Accuracy(car_data_Train['label'])
#         acc_train_m.append(car_accuracy_train)
#     acc.append(acc_m)
#     acc_train.append(acc_train_m)

# #%%

# print(pd.DataFrame(1-np.array(acc_train).T, columns=['Gini Index', 'Majority Error', 'Entropy']).to_latex())
##%% Problem 2.3
# bank_data_Train_o = pd.read_csv('bank/train.csv')
# bank_data_Test_o = pd.read_csv('bank/test.csv')
# numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
# keys = ['age', 'job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous', 'poutcome', 'y']

# vals = ((0,1),
#         ("admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"),
#         ("married","divorced","single"),
#         ("unknown","secondary","primary","tertiary"),
#         ("yes","no"),
#         (0,1),
#         ("yes","no"),
#         ("yes","no"),
#         ("unknown","telephone","cellular"),
#         (0,1),
#         ("jan", "feb", "mar","apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
#         (0,1),
#         (0,1),
#         (0,1),
#         (0,1),
#         ("unknown","other","failure","success"),
#         ("yes","no"))

# def Number_To_Binary(df, numeric_columns):

#     for col in numeric_columns:
#         median = np.median(df[col])
#         print(median)
#         large_locs = np.where(df[col] > median)[0]
#         small_locs = np.where(df[col] <= median)[0]
#         df[col].iloc[large_locs] = 1
#         df[col].iloc[small_locs] = 0
#     return df


# def Pre_proc(df, numeric_columns, fix_unknown = False):
#     df_proc = df.copy()
#     df_proc = Number_To_Binary(df_proc, numeric_columns)
#     if fix_unknown:
#         for A in df_proc.columns:
#             locs_unknown = np.where(df_proc[A] == 'unknown')[0]
#             if len(locs_unknown) == 0:
#                 continue
#             else:
#                 df_proc[A].iloc[locs_unknown] = Get_Most_Common(df_proc, A)
#     return df_proc

# bank_data_Train = Pre_proc(bank_data_Train_o, numeric_columns, fix_unknown = True)
# bank_data_Test = Pre_proc(bank_data_Test_o, numeric_columns, fix_unknown = True)

# Attribute_vals_df_bank = dict(zip(keys, vals))
# bank_accuracy = []
# bank_accuracy_train = []
# for method in range(3):
#     bank_accuracy_m = []
#     bank_accuracy_train_m = []
#     for depth in range(17):
#         print(depth)
#         DT_bank = DecisionTree(bank_data_Train, bank_data_Train.columns.to_list()[0:-1], bank_data_Train.columns.to_list()[-1], Attribute_vals_df=Attribute_vals_df_bank, depth=depth, method=method)
#         DT_bank.Train()
        
#         bank_y = DT_bank.Predict(bank_data_Test)
#         bank_accuracy_m.append(DT_bank.Accuracy(bank_data_Test['y']))
        
#         bank_y_train = DT_bank.Predict(bank_data_Train)
#         bank_accuracy_train_m.append(DT_bank.Accuracy(bank_data_Train['y']))
#     bank_accuracy.append(bank_accuracy_m)
#     bank_accuracy_train.append(bank_accuracy_train_m)
    
# #%%
# import matplotlib.pyplot as plt
# plt.plot(range(17), 1-np.array(bank_accuracy_train).T)
# plt.legend(['Gini Index', 'Majority Error', 'Entropy'])
# plt.xlabel('depth')
# plt.ylabel('accuracy')

# print(pd.DataFrame(1-np.array(bank_accuracy_train).T, columns=['Gini Index', 'Majority Error', 'Entropy']).to_latex())