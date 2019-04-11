import numpy as np
from sklearn import tree
from utils.dataLoader import dataLoader

class DT(object):
    '''
    A decision tree classifier using sklearn.tree.DecisionTreeClassifier    
    '''
    def __init__(self):
        pass
    
    def train(self, X, y, max_depth):
        '''
        Train the decision tree classifier.
        
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train, ) containing the training labels,
          where y[i] is the label for X[i].
        - max_depth: the maximum depth of the tree.
        '''
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X, y)
        self.clf = clf
    
    def predict(self, X):
        '''
        Predict labels using the trained DT classifier.
        
        Inputs:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        
        Returns:
        - y: A numpy array of shape (num_test, ) containing the predicted labels,
          where y[i] is the label for X[i].
        '''
        clf = self.clf
        return clf.predict(X)
