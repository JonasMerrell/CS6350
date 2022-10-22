import numpy as np
import sys
#sys.path.append('Decision_tree')
from Decision_tree.Decision_Tree import DecisionTree
import pickle

class Boost:
    def __init__(self, df, T, labels, Attribute_vals_df, method, Y, n, depth):
        self.T = T
        self.df = df
        self.labels = labels
        self.Attribute_vals_df = Attribute_vals_df
        self.method = method
        self.Y = Y
        self.n = n
        self.depth = depth
        self.accuracy = np.zeros(T)
        self.h = np.zeros([T, len(df)])
        self.h_test = np.zeros([T, len(df)])
        self.H = np.zeros([T, len(df)])
        self.models = []
        self.alpha = np.zeros(T)
        self.error = np.zeros(T)
        Dt_1 = np.ones(len(df))/len(df)
        self.D = np.zeros([T+1, len(df)])
        self.D[0] = Dt_1
        self.data = None
        
   
    @staticmethod
    def Alpha(error):
        at = 0.5 * np.log((1 - error)/error)
        return at
    @staticmethod
    def Error(yh, Dt):
        error = 0.5 - 0.5 * np.sum(Dt*yh)
        return error
    
    def RandTreeLearn(self, replacement=True, W=None, randomforest=False, num_Attributes=None):
            data = self.df.sample(self.n, replace=replacement, ignore_index=True, weights=W)
            DT = DecisionTree(data, list(self.Attribute_vals_df.keys()), self.labels, Attribute_vals_df=self.Attribute_vals_df, method=self.method, depth=self.depth)
            DT.random_forest = randomforest
            DT.num_Attributes = num_Attributes
            DT.Train()
            return DT
    
    def AdaBoost(self, replacement=True, test_data=None): 
        if test_data is not None:
            self.init_test_vals(test_data)
        for t in range(self.T):
            if t == 0:
                data = self.df
                DT = DecisionTree(data, list(self.Attribute_vals_df.keys()), self.labels, Attribute_vals_df=self.Attribute_vals_df, method=self.method, depth=self.depth)
                DT.Train()
            else:
                DT = self.RandTreeLearn(W=self.D[t])

            self.h[t] = DT.Predict(self.df)

                
            yh = self.Y*self.h[t]
            self.error[t] = self.Error(yh, self.D[t])
            self.alpha[t] = self.Alpha(self.error[t])
            self.D[t+1] = self.D[t] * np.exp(-self.alpha[t] * yh)
            self.models.append(DT)
            
            self.H[t] = np.sign(self.alpha[0:t+1] @ self.h[0:t+1,:])
            self.accuracy[t] = np.sum(self.Y*self.H[t] == 1)/len(self.df)
            
            if test_data is not None:
                self.h_test[t] = DT.Predict(test_data)
                yh_test = self.Y_test*self.h_test[t]
                self.error_test[t] = self.Error(yh_test, self.D[t])        
                self.H_test[t] = np.sign(self.alpha[0:t+1] @ self.h_test[0:t+1,:])
                self.accuracy[t] = np.sum(self.Y_test*self.H_test[t] == 1)/len(test_data)
        
        if test_data is not None:
            self.data = {'alpha':self.alpha, 'accuracy':self.accuracy, 'error':self.error, 'T':range(self.T), 'H':self.H, 'h':self.h, 'D':self.D, 'models':self.models}
            self.data_test = {'accuracy':self.accuracy,'error':self.error,'H':self.H_test, 'h':self.h_test}
        else:
            self.data = {'alpha':self.alpha, 'accuracy':self.accuracy, 'error':self.error, 'T':range(self.T), 'H':self.H, 'h':self.h, 'D':self.D, 'models':self.models}
    
    def get_values(self, DT, t):
        yh = self.Y*self.h[t]
        self.error[t] = self.Error(yh, self.D[0])
        self.models.append(DT)
        self.H[t] = np.sign(np.mean(self.h[0:t+1], axis=0))
        self.accuracy[t] = np.sum(self.Y*self.H[t] == 1)/len(self.df)
        
    def get_values_test(self, DT, t, test_data):
        yh = self.Y_test*self.h_test[t]
        self.error_test[t] = self.Error(yh, self.D[0])
        self.H_test[t] = np.sign(np.mean(self.h_test[0:t+1], axis=0))
        self.accuracy_test[t] = np.sum(self.Y_test*self.H_test[t] == 1)/len(test_data)
        
    def init_test_vals(self, test_data):
        self.h_test = np.zeros([self.T, len(test_data)])
        self.H_test = np.zeros([self.T, len(test_data)])
        self.accuracy_test = np.zeros(self.T)
        self.error_test = np.zeros(self.T)
        self.Y_test = test_data['y']
       
    def Bagging(self, replacement=True, test_data=None, mpi=False, randomforest=False, num_Attributes=1):
        if test_data is not None:
            self.init_test_vals(test_data)
        for t in range(self.T):
            DT = self.RandTreeLearn(replacement=replacement, randomforest=randomforest, num_Attributes=num_Attributes)
            self.h[t] = DT.Predict(self.df)
            if not mpi:
                self.get_values(DT, t)
                if test_data is not None:
                    self.h_test[t] = DT.Predict(test_data)
                    self.get_values_test(DT, t, test_data)              
            else:
                if test_data is not None:
                    self.h_test[t] = DT.Predict(test_data)

        if mpi:
            self.data = {'h':self.h}
            if test_data is not None:
                self.data_test = {'h':self.h_test}
        else:
            self.data = {'accuracy':self.accuracy, 'error':self.error, 'T':range(self.T), 'H':self.H, 'h':self.h, 'models':self.models}
            if test_data is not None:
                self.data_test = {'accuracy':self.accuracy_test, 'error':self.error_test, 'T':range(self.T), 'H':self.H_test, 'h':self.h_test}

    def Bagging_MPI(self, replacement=False, test_data=None):
        h = np.zeros([self.T, len(self.df)])
        h_test = np.zeros([self.T, len(test_data)])
        for t in range(self.T):
            DT = self.RandTreeLearn(replacement=replacement)
            h[t] = DT.Predict(self.df)
            h_test[t] = DT.Predict(test_data)
            
        return h, h_test
            

    
    def save_results(self, fname, data):
        with open(fname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    

