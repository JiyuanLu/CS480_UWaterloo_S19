import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.dataLoader import dataLoader

class KNN(object):
    '''
    A KNN classifier using sklearn.tree.KNeighborsClassifier
    '''
    def __init__(self):
        pass
    
    def train(self, X, y, n_neighbors):
        '''
        Train the KNN classifier.
        
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train, ) containing the training labels,
          where y[i] is the label for X[i].
        - n_neighbors: the number of neighbors considered for voting.
        '''
        clf = KNeighborsClassifier(n_neighbors)
        clf.fit(X, y)
        self.clf = clf
        
    def predict(self, X):
        '''
        Predict labels using the trained KNN classifier.
        
        Inputs:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        
        Returns:
        - y: A numpy array of shape (num_test, ) containing the predicted labels,
          where y[i] is the label for X[i].
        '''
        clf = self.clf
        return clf.predict(X)