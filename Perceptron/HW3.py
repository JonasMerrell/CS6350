import numpy as np
import matplotlib.pyplot as plt
def dist(w, X):
    return (w @ X.T)/np.linalg.norm(w)


#%%
X = np.array([[1, 1, 1],
              [1, -1, 1],
              [0, 0, 1],
              [-1, 3, 1],
     ])

y = np.array([1, -1, -1, 1])
w = np.array([2, 3, -4])

margin = np.min(dist(w, X)*y)

#%%

X = np.array([[-1, 0, 1],
              [0, -1, 1],
              [1, 0, 1],
              [0, 1, 1]])

y = np.array([-1, -1, 1, 1])


w = np.array([1, 1, 0])


plt.scatter(X[:,0], X[:,1], c=y)
def f(x):
    return -x
x = np.linspace(-1, 1, 5)
plt.plot(x, f(x))
margin = np.min(dist(w, X)*y)
#%%
X = np.array([[-1, 0, 1],
              [0, -1, 1],
              [1, 0, 1],
              [0, 1, 1]])

y = np.array([-1, 1, -1, 1])

plt.scatter(X[:,0], X[:,1], c=y)







