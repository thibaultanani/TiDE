import abc
import os

import numpy as np
import psutil

from feature_selections import FeatureSelection
from utility import fitness


class Heuristic(FeatureSelection):
    """
    Parent class for all heuristics

    Args:
        N (int)   : The number of individuals evaluated per iteration
    """
    def __init__(self, name, target, model, train, test=None, k=None, standardisation=None, drops=None, metric=None,
                 N=None, Gmax=None, Tmax=None, ratio=None, suffix=None, verbose=None):
        super().__init__(name, target, model, train, test, k, standardisation, drops, metric, Gmax, Tmax, ratio,
                         suffix, verbose)
        self.N = N or 50

    @abc.abstractmethod
    def start(self, pid):
        pass

    @staticmethod
    def pprint_(print_out, name, pid, maxi, best, mean, feats, time_exe, time_total, entropy, g, cpt, verbose):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}" \
                  "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}    entropy : {:2.3f}" \
            .format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt, entropy)
        print_out = print_out + display
        if verbose: print(display)
        return print_out

    @staticmethod
    def sprint_(print_out, name, pid, maxi, best, mean, feats, time_exe, time_total, g, cpt, verbose):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}" \
                  "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}" \
            .format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt)
        print_out = print_out + display
        if verbose: print(display)
        return print_out

    def save(self, name, bestInd, g, t, last, specifics, out):
        a = os.path.join(os.path.join(self.path, 'results.txt'))
        f = open(a, "w")
        methods = [self.model[m].__class__.__name__ for m in range(len(self.model))]
        bestSubset = [self.cols[i] for i in range(len(self.cols)) if bestInd[i]]
        score_train, y_true, y_pred = fitness(train=self.train, test=self.test, columns=self.cols, ind=bestInd,
                                              target=self.target, models=self.model, metric=self.metric,
                                              standardisation=self.standardisation, ratio=0, k=self.k)
        tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
        if isinstance(bestInd, np.ndarray):
            bestInd = bestInd.tolist()
        string = "Heuristic: " + name + os.linesep + \
                 "Population: " + str(self.N) + os.linesep + \
                 "Generation: " + str(self.Gmax) + os.linesep + \
                 "Generation Performed: " + str(g) + os.linesep + \
                 "Latest Improvement: " + str(last) + os.linesep + \
                 "K-fold cross validation: " + str(self.k) + os.linesep + \
                 "Standardisation: " + str(self.standardisation) + os.linesep + \
                 specifics + \
                 "Methods List: " + str(methods) + os.linesep + \
                 "Best Method: " + str(self.model[bestInd[-1]].__class__.__name__)+ os.linesep + \
                 "Best Score: " + str(score_train) + os.linesep + "TP: " + str(tp) + \
                 " TN: " + str(tn) + " FP: " + str(fp) + " FN: " + str(fn) + os.linesep + \
                 "Best Subset: " + str(bestSubset) + os.linesep + \
                 "Number of Features: " + str(len(bestSubset)) + os.linesep + \
                 "Execution Time: " + str(round(t.total_seconds())) + " (" + str(t) + ")" + os.linesep + \
                 "Memory: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path, 'log.txt'))
        f = open(a, "a")
        f.write(out)
