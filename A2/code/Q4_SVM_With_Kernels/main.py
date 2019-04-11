
# coding: utf-8

# In[1]:


import numpy as np
from utils.LoadDataset import LoadDataset
from sklearn.svm import SVC

X, y = LoadDataset()


# In[2]:


# Linear Kernel
print('Linear Kernel:')
C_choices = [1, 10, 100]
C_to_accuracy = {}
for C in C_choices:
    clf = SVC(C=C, kernel='linear')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    accuracy = clf.score(X, y)
    C_to_accuracy[C] = accuracy

for C in sorted(C_to_accuracy.keys()):
    print('C = %d, accuracy = %.5f' % (C, C_to_accuracy[C]))


# In[3]:


# Polynomial Kernel
print('Polynomial Kernel:')
C_choices = [1, 10, 100]
degree_choices = [3, 4, 5]
C_degree_to_accuracy = {}
for C in C_choices:
    for d in degree_choices:
        clf = SVC(C=C, kernel='poly', degree=d)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        accuracy = clf.score(X, y)
        C_degree_to_accuracy[(C, d)] = accuracy

for C, d in sorted(C_degree_to_accuracy.keys()):
    print('C = %d, degree = %d, accuracy = %.5f' % (C, d, C_degree_to_accuracy[(C, d)]))


# In[4]:


# RBF Kernel
print('RBF Kernel:')
num_features = X.shape[1]
C_choices = [1, 10, 100]
gamma_choices = [0.2 / num_features, 1.0 / num_features, 5.0 / num_features]
C_gamma_to_accuracy = {}
for C in C_choices:
    for g in gamma_choices:
        clf = SVC(C=C, kernel='rbf', gamma=g)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        accuracy = clf.score(X, y)
        C_gamma_to_accuracy[(C, g)] = accuracy
        
for C, g in sorted(C_gamma_to_accuracy.keys()):
    print('C = %d, gamma = %.5f, accuracy = %.5f' % (C, g, C_gamma_to_accuracy[(C, g)]))

