import numpy as np

def LoadDataset():
    '''
    Load the training data.
    
    Returns:
    - X_train: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y_train: A numpy array of shape (num_train, ) containing the training labels,
      where y_train[i] is the label for X_train[i].
    '''
    X_train = np.loadtxt('data/ionosphere.csv', delimiter=',', usecols=range(34), dtype='float')
    y_train = np.loadtxt('data/ionosphere.csv', delimiter=',', usecols=34, dtype='str')
    return X_train, y_train
    
    