import numpy as np

class KNN(object):
    ''' A KNN classifier with L2 distance '''
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        '''
        Train the classifier. For knn, just memorize the training data.
        
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train, ) containing the training labels,
          where y[i] is the label for X[i].
        '''
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1):
        '''
        Predict labels for test data using this classifier.
        
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
          of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        
        Returns:
        - y: A numpy array of shape (num_test, ) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        '''
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)
    
    def compute_distances(self, X):
        '''
        Compute the distance between each test point in X and each training point
        in self.X_train.
        
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j] is
          the Euclidean distance between the ith test point and the jth training
          point.
        '''
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        matrix_product = np.matmul(X, self.X_train.T)
        test_sum = np.sum(np.square(X), axis=1)
        train_sum = np.sum(np.square(self.X_train), axis=1)
        dists = np.sqrt(test_sum.reshape(-1, 1) - 2 * matrix_product + train_sum)
        
        return dists
    
    def predict_labels(self, dists, k=1):
        '''
        Given a matrix of distances between test points and training points, 
        predict a label for each test point.
        
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.
          
        Returns:
        - y: A numpy array of shape (num_test, ) containing prediected labels for
          the test data, where y[i] is the predicted label for the test point X[i].
        '''
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to 
            # the ith test point.
            closest_y = []           
            y_indices = np.argsort(dists[i, :])
            closest_y = self.y_train[y_indices[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
        
        
        
    
    
    
    
    