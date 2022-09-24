import numpy as np
import pandas as pd

class Node(object):
    def __init__(self):
        self.Attribute = None
        self.values = []
        self.branches = []
        self.depth = 0

    def get_name(self):
        return 'Node'

class Leaf(object):
    def __init__(self):
        self.Attribute = None
        self.Attribute_val = None
        self.value = None

    def get_name(self):
        return 'Leaf'
    

class DecisionTree():
    def __init__(self, data, Attributes, Label, Attribute_vals_df=None, depth=10, method=0):
        self.S = data
        self.Attributes = np.array(Attributes)
        self.L = Label
        self.Attribute_vals_df = Attribute_vals_df
        self.Tree = None
        self.y = []
        self.method = method
        self.depth = depth
        self.method_name = None
        self.int_E = None
        return
    
    def Get_Method(self):
        if self.method == 0:
            self.method_name = 'Gini Index'
            return self.GiniIndex
        elif self.method == 1:
            self.method_name = 'Majority Error'
            return self.MajorityError
        elif self.method == 2:
            self.method_name = 'Entropy'
            return self.Entropy
        else:
            return

    def Entropy(self, data, A, v):
        P, s = self.Prob(data, A, v)
        H = 0
        for p in P:
            if p == 0:
                H -= 0
            else:
                H -= p*np.log2(p)
                
        #print(f'Entropy: =  {H}\\\\')
        return H, s

    def Prob(self, data, A, v):
        locs = np.where(data[A] == v)[0]
        Lv = data[self.L].iloc[locs]
        P = []
        s = len(Lv)
        #print(f'total num: {s}\\\\')
        for i in np.unique(data[self.L]):
            #print(f'num for {i}: {np.count_nonzero(Lv == i)}\\\\')
            P.append(np.count_nonzero(Lv == i)/s)

        return P, s
    
    def MajorityError(self, data, A, v):
        locs = np.where(data[A] == v)[0]
        Lv = data[self.L].iloc[locs]
        count = []
        s = len(Lv)
        #print(f'total num: {s}\\\\')
        for i in np.unique(data[self.L]):
            #print(f'num for {i}: {np.count_nonzero(Lv == i)}\\\\')
            count.append(np.count_nonzero(Lv == i))
        min_val = np.min(count)
        ME = min_val / sum(count)
        #print(ME)
        
        #print(f'Majority Error: {min_val}/{sum(count)} =  {ME}\\\\')
        return ME, s
    
    def GiniIndex(self, data, A, v):
        P, s = self.Prob(data, A, v) 
        S = 0
        for p in P:
            S += p**2 
        GI = 1 - S
        #print(f'Gini Index: =  {GI}\\\\')
        return GI, s
    
    
    def InfoGain(self, data, A, method=None):
        if self.int_E == None:
            self.init_E = method(data, self.L, data[self.L])[0]
            #print(f'intial {self.method_name}: {self.init_E}\\\\')
        V = np.unique(data[A])
        Ps = []
        ss = []
        expected_E = 0
        #print(f'Attribute: {A}\\\\')
        for v in V:
            #print(f'Value: {v}\\\\')
            E, s = method(data, A, v)
            
            expected_E += s/len(data[self.L])*E
            #print(f'expected {self.method_name}: {s}/{len(data[self.L])} * E = {s/len(data[self.L])*E}')
        IG = self.init_E - expected_E
        #print(f'Gain for {A}: {IG}\\\\')
        return IG

    def Train(self, method = 'ID3'):
        if method == 'ID3':
            self.Tree = Node()
            self.Tree.Attribute = 'root'
            self.ID3(self.S, self.Attributes, curr_branch=self.Tree)
        return

    def ID3(self, S, Attributes, curr_A=None, curr_v=None, curr_branch=None):
        #print('Current Subset of dataset\\\\')
        #print(S.to_latex())
        if len(np.unique(S[self.L])) == 1:
            
            curr_branch.branches.append(Leaf())
            print(f'add leaf: {S[self.L].iloc[0]}\\\\')
            print('\\\\')
            curr_branch.branches[-1].value = S[self.L].iloc[0]
            #curr_branch = curr_node
            return S[self.L].iloc[0]
        else:
            InformationGain = []
            for A in Attributes:
                InformationGain.append(self.InfoGain(S, A, method=self.Get_Method()))
            for i, a in enumerate(Attributes):
                print(f'Attribute: {a}, Information Gain: {InformationGain[i]}\\\\')
            A = Attributes[np.argmax(InformationGain)]
            print(f'best Attribute: {A}\\\\')
            print('\\\\')
            
            curr_branch.branches.append(Node())
            curr_branch.branches[-1].depth = curr_branch.depth 
            curr_branch = curr_branch.branches[-1]
            curr_branch.Attribute = A
            curr_branch.depth += 1
            
            try:
                Attribute_vals = self.Attribute_vals_df[A]
            except:
                Attribute_vals = np.unique(S[A])
            
            for v in Attribute_vals:
                curr_branch.values = self.Attribute_vals_df[A]
                print(f'Value: {v}\\\\')
                Sv = S.iloc[np.where(S[A] == v)[0],:]
                if np.shape(Sv)[0] == 0:
                    print('No training data for Attribute: {curr_A}, value: {v}\\\\')
                    print(f'most common label is {self.Get_Most_Common(S)}\\\\')
                    print(f'add leaf: {self.Get_Most_Common(S)}\\\\')
                    curr_branch.branches.append(Leaf())
                    curr_branch.branches[-1].value = self.Get_Most_Common(S)

                    
                elif curr_branch.depth >= self.depth:
                    curr_branch.branches.append(Leaf())
                    curr_branch.branches[-1].value = self.Get_Most_Common(Sv)
                else:
                    self.ID3(Sv, Attributes[Attributes != A], A, v, curr_branch)
                    
    def Get_Most_Common(self, Sv):
        unique, counts = np.unique(Sv[self.L], return_counts=True)
        value = unique[np.argmax(counts)]
        return value
    
    def _Predict(self, X):
        curr_branch = self.Tree
        self.Eval_Tree(X, curr_branch)
    
    def Predict(self, df):
        self.y = []
        for index, row in df.iterrows():
            self._Predict(row)
        return self.y
    
    def Eval_Tree(self, x, curr_branch):
        if curr_branch.get_name() == 'Node':
            if curr_branch.Attribute == 'root':
                curr_branch = curr_branch.branches[0]
            A = curr_branch.Attribute
            val = x[A]
            branch = np.where(np.array(curr_branch.values) == val)[0][0]
            curr_branch = curr_branch.branches[branch] 
            self.Eval_Tree(x, curr_branch)
            
        elif curr_branch.get_name() == 'Leaf':
            
            self.y.append(curr_branch.value)
            #return self.y
            
        else:
            raise ValueError 
            
    def Accuracy(self, y):
        num_True = y.to_numpy() == np.array(self.y)
        
        try:
            return np.count_nonzero(np.array(num_True) == True)/len(num_True)
        except:
            if num_True == True:
                return 1
            elif num_True == False:
                return 0



    
