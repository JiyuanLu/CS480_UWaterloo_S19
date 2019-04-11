import numpy as np
from utils.dataLoader import dataLoader
from models.DT import DT
from models.KNN import KNN

def Test(classifier, best_k):
    '''
    Use the best k attained from cross validation to train a classifier on the whole
    training dataset, then test of the test dataset.
    
    Inputs:
    - classifier: the classifier to use, DT (Decision Tree) or KNN (K Nearest Neighbors).
    - best_k: hyper-parameter for the classifier, i.e. k for KNN, maximum tree depth for DT.
    '''
    if classifier == 'DT':
        clf = DT()
    elif classifier == 'KNN':
        clf = KNN()
        
    loader = dataLoader()
    X_train, y_train, X_test, y_test = loader.loadData()
    
    clf.train(X_train, y_train, best_k)
    y_pred = clf.predict(X_test)
    num_test = y_test.shape[0]
    
    all_N = np.copy(y_test)
    all_N[:] = 'B'
    all_P = np.copy(y_test)
    all_P[:] = 'M'

    term_match = (y_pred == y_test)
    term_mismatch = np.logical_not(term_match)
    term_P = (y_pred == all_P)
    term_N = (y_pred == all_N)

    term_TP = np.logical_and(term_match, term_P)
    term_TN = np.logical_and(term_match, term_N)
    term_FP = np.logical_and(term_mismatch, term_P)
    term_FN = np.logical_and(term_mismatch, term_N)
    
    TP = np.sum(term_TP)
    TN = np.sum(term_TN)
    FP = np.sum(term_FP)
    FN = np.sum(term_FN)

    assert (TP + TN + FP + FN) == num_test

    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    accuracy = float(TP + TN) / (TP + TN + FP + FN)
    sensitivity = float(TP) / (TP + FN)
    specificity = float(TN) / (TN + FP)
    
    return precision, recall, accuracy, sensitivity, specificity
    