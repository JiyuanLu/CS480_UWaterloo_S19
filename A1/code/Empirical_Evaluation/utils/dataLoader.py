import numpy as np

class dataLoader(object):
    ''' A dataloader to load the WDBC dataset. '''
    def __init__(self):
        pass
    
    def loadData(self, train_path='data/wdbc-train.csv', test_path='data/wdbc-test.csv'):
        '''
        Inputs:
        - train_path: relative or absolute path of the training dataset.
        - test_path: relative or absolute path of the test dataset.
        
        Returns:
        - X_train: a numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: a numpy array of shape (num_train,) containing the training labels,
          where y_train[i] is the label for X_train[i].
        - X_test: a numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        - y_test: a numpy array of shape (num_test,) containing the test labels,
          where y_test[i] is the label for X_test[i].
        '''
        self.X_train = np.loadtxt(train_path, delimiter=',', usecols=range(2, 32))
        self.y_train = np.loadtxt(train_path, delimiter=',', usecols=1, dtype='str')
        self.X_test = np.loadtxt(test_path, delimiter=',', usecols=range(2, 32))
        self.y_test = np.loadtxt(test_path, delimiter=',', usecols=1, dtype='str')

        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def shuffleData(self):
        '''
        Shuffle the training data randomly before splitting into groups to be used 
        for cross validation.
        '''  
        y_train = self.y_train.reshape(-1, 1)
        X_train = self.X_train
        train_data = np.hstack((X_train, y_train))
        np.random.shuffle(train_data)
        shuffled_X_train = train_data[:, :30]
        shuffled_y_train = train_data[:, -1]
        
        return shuffled_X_train, shuffled_y_train, self.X_test, self.y_test
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                