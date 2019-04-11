import numpy as np

def LoadDataset(i, N):
    '''
    Load the training data and the validation data.
    
    Inputs:
    - i: index of the validation set you want to choose
    - N: number of folds that defines N-fold cross validation
    
    Returns:
    - trainingData: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - trainingLabel: A numpy array of shape (num_train, ) containing the training labels,
      where trainingLabel[i] is the label for trainingData[i].
    - validationData: A numpy array of shape (num_val, D) containing the validation data
      consisting of num_val samples each of dimension D.
    - validationLabel: A numpy array of shape (num_val, ) containing the validation labels,
      where validationLabel[i] is the label for validationData[i].
    '''
    flag = True # turn false after we encounter the first training set
    for j in range(i, i+N):
        # construct training and validation set
        data = np.loadtxt('data/fData' + str(j%N) + '.csv', delimiter=',', dtype='float')
        label = np.loadtxt('data/fLabels' + str(j%N) + '.csv', delimiter=',', dtype='float')
        if i == j:
            validationData = data
            validationLabel = label
        elif flag:
            trainingData = data
            trainingLabel = label
            flag = False
        else:
            trainingData = np.concatenate((trainingData, data))
            trainingLabel = np.concatenate((trainingLabel, label))
    return trainingData, trainingLabel, validationData, validationLabel