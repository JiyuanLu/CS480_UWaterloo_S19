import numpy as np
import scipy


class LinearRegression(object):
    ''' linear least square regression with squared l2-norm penalty (Ridge Regression) '''\
    
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.w = None
    
    def fit(self, X, y):
        ''' use the closed form solution. '''
        alpha = self.alpha
        num_train, D = X.shape
        if self.fit_intercept:
            all_ones = np.ones(num_train).reshape(-1, 1)
            X = np.hstack((all_ones, X))
            D = D + 1      
        identity = np.identity(D)
        #identity[0][0] = 0
        first_term = np.linalg.inv(np.matmul(X.T, X) + alpha * identity)
        second_term = np.matmul(X.T, y)
        self.w = np.matmul(first_term, second_term)
        
    def predict(self, X):
        num_test = X.shape[0]
        if self.fit_intercept:
            all_ones = np.ones(num_test).reshape(-1, 1)
            X = np.hstack((all_ones, X))
        return np.matmul(X, self.w)