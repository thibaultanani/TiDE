import os
import time
from datetime import timedelta

import numpy as np
import psutil
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif, mutual_info_classif
from skrebate import SURF

from feature_selections import FeatureSelection
from utility import createDirectory, fitness


class Filter(FeatureSelection):
    """
    Parent class for filter methods

    Args:
        method (str)  : Filter method to use for feature selection
        quantitative (bool) : If the target feature is quantitative or not (only for mrmr)
    """
    def __init__(self, name, target, model, train, test=None, k=None, standardisation=None, drops=None, metric=None,
                 Gmax=None, quantitative=None, Tmax=None, ratio=None, suffix=None, verbose=None, method=None):
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
        elif method == "SURF":
            self.method, self.func, self.v_name = "SURF", self.surf_selection, "SURF"
        self.quantitative = quantitative or None
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

    def save(self, name, bestInd, scores, g, t, last, out):
        a = os.path.join(os.path.join(self.path, 'results.txt'))
        f = open(a, "w")
        bestSubset = [self.cols[i] for i in range(len(self.cols)) if bestInd[i]]
        score_train, y_true, y_pred = fitness(train=self.train, test=self.test, columns=self.cols, ind=bestInd,
                                              target=self.target, model=self.model, metric=self.metric,
                                              standardisation=self.standardisation, ratio=0, k=self.k)
        tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
        string = "Filter: " + str(name) + os.linesep + \
                 "Iterations: " + str(self.Gmax) + os.linesep + \
                 "Iterations Performed: " + str(g) + os.linesep + \
                 "Latest Improvement: " + str(last) + os.linesep + \
                 "Latest Improvement (Ratio): " + str(1 - (last / g)) + os.linesep + \
                 "K-fold cross validation: " + str(self.k) + os.linesep + \
                 "Standardisation: " + str(self.standardisation) + os.linesep + \
                 "Method: " + str(self.model.__class__.__name__) + os.linesep + \
                 "Best Score: " + str(score_train) + os.linesep + "TP: " + str(tp) + \
                 " TN: " + str(tn) + " FP: " + str(fp) + " FN: " + str(fn) + os.linesep + \
                 "Best Subset: " + str(bestSubset) + os.linesep + \
                 "Number of Features: " + str(len(bestSubset)) + os.linesep + \
                 "Number of Features (Ratio): " + str(len(bestSubset) / len(self.cols)) + os.linesep + \
                 "Score of Features: " + str(scores) + os.linesep + \
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
                coef, _ = pearsonr(df[column], df[target].values)
                corr_results.append((column, coef))
        corr_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in corr_results]
        return sorted_features, corr_results

    @staticmethod
    def anova_selection(df, target):
        X = df.drop([target], axis=1)
        y = df[target]
        f_values, _ = f_classif(X, y)
        f_results = list(zip(X.columns, f_values))
        f_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in f_results]
        return sorted_features, f_results

    @staticmethod
    def mutual_info_selection(df, target):
        X = df.drop([target], axis=1)
        y = df[target]
        mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        mi_results = list(zip(X.columns, mi_scores))
        mi_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in mi_results]
        return sorted_features, mi_results

    @staticmethod
    def mrmr_selection(df, target, is_quantitative=False):
        X = df.drop([target], axis=1)
        y = df[target]
        if is_quantitative:
            relevance, pearson_values = {}, []
            for column in X.columns:
                coef, _ = pearsonr(X[column], y)
                pearson_values.append(coef)
                relevance[column] = coef
            min_relevance, max_relevance = min(pearson_values), max(pearson_values)
            relevance = {column: (value - min_relevance) / (max_relevance - min_relevance)
                         for column, value in relevance.items()}
        else:
            f_values, _ = f_classif(X, y)
            # relevance = {column: f_value for column, f_value in zip(X.columns, f_values)}
            min_f_value, max_f_value = min(f_values), max(f_values)
            relevance = {column: (f_value - min_f_value) / (max_f_value - min_f_value)
                         for column, f_value in zip(X.columns, f_values)}
        correlation_matrix = X.corr()
        min_corr, max_corr = correlation_matrix.min().min(), correlation_matrix.max().max()
        normalised_corr_matrix = (correlation_matrix - min_corr) / (max_corr - min_corr)
        S, R, mrmr_scores_list = [], list(X.columns), {}
        first_feature = max(relevance, key=relevance.get)
        S.append(first_feature), R.remove(first_feature)
        mrmr_scores_list[first_feature] = relevance[first_feature]
        while R:
            mrmr_scores = {}
            for feature in R:
                redundancy = normalised_corr_matrix.loc[feature, S].mean()
                mrmr_scores[feature] = relevance[feature] - redundancy
            next_feature = max(mrmr_scores, key=mrmr_scores.get)
            S.append(next_feature)
            R.remove(next_feature)
            mrmr_scores_list[next_feature] = mrmr_scores[next_feature]
        return S, mrmr_scores_list

    @staticmethod
    def surf_selection(df, target):
        X = df.drop([target], axis=1).values.astype('float64')
        y = df[target].values
        surf = SURF(n_features_to_select=X.shape[1], discrete_threshold=2, n_jobs=-1)
        surf.fit(X, y)
        surf_scores = surf.feature_importances_
        surf_results = list(zip(df.columns, surf_scores))
        surf_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in surf_results]
        return sorted_features, surf_results

    def start(self, pid):
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        score, model, col, vector, s = -np.inf, [], [], [], -np.inf
        G, same, stop = 0, 0, False
        try:
            sorted_features, filter_scores = self.func(df=self.train, target=self.target)
        except TypeError:
            sorted_features, filter_scores = self.func(df=self.train, target=self.target,
                                                       is_quantitative=self.quantitative)
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
            s = fitness(train=self.train, test=self.test, columns=self.cols, ind=v,
                        target=self.target, model=self.model, metric=self.metric,
                        standardisation=self.standardisation, ratio=self.ratio, k=self.k)[0]
            if s > score:
                same = 0
                score, vector = s, v
                col = [self.cols[i] for i in range(len(self.cols)) if v[i]]
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
                self.save(name=self.method, bestInd=vector, scores=filter_scores, g=G,
                          t=timedelta(seconds=(time.time() - debut)), last=G - same, out=print_out)
                print_out = ""
                if stop:
                    break
        return score, vector, col, self.model, pid, self.v_name, G - same, G
