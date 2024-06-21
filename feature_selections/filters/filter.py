import os
import random
import time
from datetime import timedelta

import numpy as np
import psutil
from mrmr import mrmr_classif
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from skrebate import ReliefF, SURF, MultiSURF, TuRF

from feature_selections import FeatureSelection
from utility import createDirectory, fitness


class Filter(FeatureSelection):
    """
    Parent class for filter methods

    Args:
        method (str)  : Filter method to use for feature selection
    """
    def __init__(self, name, target, model, train, test=None, k=None, standardisation=None, drops=None, metric=None,
                 Gmax=None, Tmax=None, ratio=None, suffix=None, verbose=None, method=None):
        super().__init__(name, target, model, train, test, k, standardisation, drops, metric, Gmax, Tmax, ratio, suffix,
                         verbose)
        if method is None or method == "Correlation":
            self.method, self.func, self.v_name = "Correlation", self.correlation_selection, "CORR"
        elif method == "Anova":
            self.method, self.func, self.v_name = "Anova", self.anova_selection, "ANOV"
        elif method == "Mutual Information":
            self.method, self.func, self.v_name = "Mutual Information", self.mutual_info_selection, "MI  "
        elif method == "MRMR":
            self.method, self.func, self.v_name = "MRMR", self.mrmr_selection, "MRMR"
        elif method == "ReliefF":
            self.method, self.func, self.v_name = "ReliefF", self.relieff_selection, "RELI"
        elif method == "SURF":
            self.method, self.func, self.v_name = "SURF", self.surf_selection, "SURF"
        elif method == "MultiSURF":
            self.method, self.func, self.v_name = "MultiSURF", self.multisurf_selection, "MS  "
        elif method == "TURF":
            self.method, self.func, self.v_name = "TURF", self.turf_selection, "TURF"
        elif method == "RandomForest":
            self.method, self.func, self.v_name = "RandomForest", self.random_forest_selection, "RF  "
        self.path = os.path.join(self.path, str.lower(self.method) + self.suffix)
        createDirectory(path=self.path)


    @staticmethod
    def print_(print_out, name, pid, maxi, best, mean, feats, time_exe, time_total, g, cpt, verbose):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}" \
                  "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}" \
            .format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt)
        print_out = print_out + display
        if verbose: print(display)
        return print_out

    def save(self, name, bestInd, g, t, last, out):
        a = os.path.join(os.path.join(self.path, 'results.txt'))
        f = open(a, "w")
        methods = [self.model[m].__class__.__name__ for m in range(len(self.model))]
        bestSubset = [self.cols[i] for i in range(len(self.cols)) if bestInd[i]]
        score_train, y_true, y_pred = fitness(train=self.train, test=self.test, columns=self.cols, ind=bestInd,
                                              target=self.target, models=self.model, metric=self.metric,
                                              standardisation=self.standardisation, ratio=0, k=self.k)
        tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
        string = "Filter: " + str(name) + os.linesep + \
                 "Iterations: " + str(self.Gmax) + os.linesep + \
                 "Iterations Performed: " + str(g) + os.linesep + \
                 "Latest Improvement: " + str(last) + os.linesep + \
                 "K-fold cross validation: " + str(self.k) + os.linesep + \
                 "Standardisation: " + str(self.standardisation) + os.linesep + \
                 "Methods List: " + str(methods) + os.linesep + \
                 "Best Method: " + str(self.model[bestInd[-1]].__class__.__name__) + os.linesep + \
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

    @staticmethod
    def correlation_selection(df, target):
        corr_results = []
        for column in df.columns:
            if column != target:
                coef, _ = spearmanr(df[column], df[target].values)
                corr_results.append((column, coef))
        corr_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in corr_results]
        return sorted_features

    @staticmethod
    def anova_selection(df, target):
        X = df.drop([target], axis=1)
        y = df[target]
        f_values, _ = f_classif(X, y)
        f_results = list(zip(X.columns, f_values))
        f_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in f_results]
        return sorted_features

    @staticmethod
    def mutual_info_selection(df, target):
        X = df.drop([target], axis=1)
        y = df[target]
        mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        mi_results = list(zip(X.columns, mi_scores))
        mi_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in mi_results]
        return sorted_features

    @staticmethod
    def mrmr_selection(df, target):
        X = df.drop([target], axis=1)
        y = df[target]
        sorted_features = mrmr_classif(X=X, y=y, K=X.shape[1])
        return sorted_features

    @staticmethod
    def relieff_selection(df, target):
        X = df.drop([target], axis=1).values.astype('float64')
        y = df[target].values
        relief = ReliefF(n_neighbors=100, n_features_to_select=X.shape[1])
        relief.fit(X, y)
        rel_scores = relief.feature_importances_
        rel_results = list(zip(df.columns, rel_scores))
        rel_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in rel_results]
        return sorted_features

    @staticmethod
    def surf_selection(df, target):
        X = df.drop([target], axis=1).values.astype('float64')
        y = df[target].values
        surf = SURF(n_features_to_select=X.shape[1])
        surf.fit(X, y)
        surf_scores = surf.feature_importances_
        surf_results = list(zip(df.columns, surf_scores))
        surf_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in surf_results]
        return sorted_features

    @staticmethod
    def multisurf_selection(df, target):
        X = df.drop([target], axis=1).values.astype('float64')
        y = df[target].values
        multisurf = MultiSURF(n_features_to_select=X.shape[1])
        multisurf.fit(X, y)
        multi_scores = multisurf.feature_importances_
        multi_results = list(zip(df.columns, multi_scores))
        multi_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in multi_results]
        return sorted_features

    @staticmethod
    def turf_selection(df, target):
        headers = list(df.drop([target], axis=1))
        X = df.drop([target], axis=1).values.astype('float64')
        y = df[target].values
        turf = TuRF(core_algorithm="ReliefF", n_features_to_select=X.shape[1])
        turf.fit(X, y, headers=headers)
        turf_scores = turf.feature_importances_
        turf_results = list(zip(df.columns, turf_scores))
        turf_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in turf_results]
        return sorted_features

    @staticmethod
    def random_forest_selection(df, target):
        X = df.drop([target], axis=1)
        y = df[target]
        rf = RandomForestClassifier(n_estimators=1000, random_state=42)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        rf_results = list(zip(X.columns, rf_scores))
        rf_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in rf_results]
        return sorted_features

    def start(self, pid):
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        score, model, col, vector, s = -np.inf, [], [], [], -np.inf
        G, same, stop = 0, 0, False
        sorted_features = self.func(df=self.train, target=self.target)
        k = [val for val in range(1, 101)]
        num_features = [int(round(len(sorted_features) * (val / 100.0))) for val in k]
        num_features = [val for val in num_features if val >= 1]
        num_features = list(dict.fromkeys(num_features))
        while G < self.Gmax:
            instant = time.time()
            top_k_features = sorted_features[:num_features[G]]
            G = G + 1
            same = same + 1
            v = [0] * self.D
            for var in top_k_features:
                v[self.cols.get_loc(var)] = 1
            for model_index in range(len(self.model)):
                v_with_model = v.copy()
                v_with_model.append(model_index)
                s = fitness(train=self.train, test=self.test, columns=self.cols, ind=v_with_model,
                            target=self.target, models=self.model, metric=self.metric,
                            standardisation=self.standardisation, ratio=self.ratio, k=self.k)[0]
                if s > score:
                    same = 0
                    score, vector = s, v_with_model
                    col = [self.cols[i] for i in range(len(self.cols)) if v_with_model[i]]
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            print_out = self.print_(print_out=print_out, name=self.v_name, pid=pid, maxi=score, best=s, mean=s,
                                    feats=len(col), time_exe=time_instant, time_total=time_debut, g=G, cpt=same,
                                    verbose=self.verbose) + "\n"
            # If the time limit is exceeded, we stop or 100% reached
            if time.time() - debut >= self.Tmax or G == len(num_features):
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                self.save(name=self.method, bestInd=vector, g=G, t=timedelta(seconds=(time.time() - debut)),
                          last=G - same, out=print_out)
                print_out = ""
                if stop:
                    break
        return score, vector, col, self.model[vector[-1]], pid, self.v_name, G - same, G

