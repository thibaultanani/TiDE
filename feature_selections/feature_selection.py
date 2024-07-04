import os
import numpy as np
import abc
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


class FeatureSelection:
    """
    Parent class for all feature selection methods (filter, wrapper and heuristic)

    Args:
        name (str): Results folder name
        target (str): Target feature name
        model (list): List of sklearn learning method objects
        train (pd.DataFrame): Training data
        test (pd.DataFrame, None): Testing data
        k (int): Number of folds for validation if no test data is provided
        standardisation (bool): Whether to standardise the data or not
        drops (list): Features to drop before execution
        metric: Metric to optimize between accuracy, precision and recall
        Gmax (int): Total number of iterations
        Tmax (int): Total number of seconds allocated before shutdown
        ratio (float): Importance of the number of features selected in relation to the score calculated
        suffix (str): Suffix in the folder name of a method (Important when lauching twice the same method !)
        verbose (bool): Print progress
    """
    def __init__(self, name, target, model, train, test=None, k=None, standardisation=None, drops=None, metric=None,
                 Gmax=None, Tmax=None, ratio=None, suffix=None, verbose=None):
        drops = drops or []
        self.train = train.drop(drops, axis=1)
        if isinstance(test, pd.DataFrame):
            self.test = test.drop(drops, axis=1)
        else:
            self.test = None
        self.name = name
        self.target = target
        self.cols = self.train.drop([target], axis=1).columns
        unique, count = np.unique(train[target], return_counts=True)
        self.n_class = len(unique)
        self.D = len(self.cols)
        self.metric = metric or balanced_accuracy_score
        self.model = model
        self.k = k or 10
        self.Gmax = Gmax or 1000000
        self.Tmax = Tmax or 3600
        self.ratio = ratio or 0.00001
        self.suffix = suffix or ''
        self.verbose = verbose or True
        self.standardisation = standardisation or False
        self.path = os.path.join(os.getcwd(), os.path.join('out', self.name))
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @abc.abstractmethod
    def start(self, pid):
        pass

    @staticmethod
    def calculate_confusion_matrix_components(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        return TP, TN, FP, FN
