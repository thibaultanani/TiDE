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
        pipeline (Pipeline): sklearn pipeline object including preprocessing + model
        train (pd.DataFrame): Training data
        test (pd.DataFrame, None): Testing data
        cv (CV): Cross validation object if no test data is provided
        drops (list): Features to drop before execution
        scoring: Metric to optimize between accuracy, precision and recall
        Gmax (int): Total number of iterations
        Tmax (int): Total number of seconds allocated before shutdown
        ratio (float): Importance of the number of features selected in relation to the score calculated
        suffix (str): Suffix in the folder name of a method (Important when launching twice the same method!)
        verbose (bool): Print progress
        output (str): The name of the output folder ('out' by default)
    """
    def __init__(self, name, target, pipeline, train, test=None, cv=None, drops=None, scoring=None,
                 Gmax=None, Tmax=None, ratio=None, suffix=None, verbose=None, output=None):
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
        self.scoring = scoring or balanced_accuracy_score
        self.pipeline = pipeline
        self.cv = cv
        self.Gmax = Gmax or 1000000
        self.Tmax = Tmax or 3600
        self.ratio = ratio if ratio is not None else 0.00001
        self.suffix = suffix or ''
        self.verbose = verbose or True
        if output is not None:
            self.path = os.path.join(os.getcwd(), os.path.join(output, self.name))
        else:
            self.path = os.path.join(os.getcwd(), os.path.join('out', self.name))
        if not os.path.exists(self.path):
            print(self.path)
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
