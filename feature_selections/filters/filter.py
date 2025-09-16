import os
import time
from datetime import timedelta

import numpy as np
import psutil
import joblib
from scipy.stats import pearsonr
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from feature_selections import FeatureSelection
from utility import createDirectory, fitness


class Filter(FeatureSelection):
    """
    Parent class for filter methods

    Args:
        method (str): Filter method to use for feature selection (default: 'Pearson Correlation')
    """
    def __init__(self, name, target, pipeline, train, test=None, cv=None, drops=None, scoring=None, Gmax=None,
                 Tmax=None, ratio=None, suffix=None, verbose=None, output=None, method=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring,
                         Gmax, Tmax, ratio, suffix, verbose, output)
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
        if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
            self.quantitative = False
        else:
            self.quantitative = True
        self.path = os.path.join(self.path, str.lower(self.method) + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def print_(print_out, name, pid, maxi, best, mean, feats, time_exe, time_total, g, cpt, verbose):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}" \
                  "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}" \
            .format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt)
        print_out = print_out + display
        if verbose:
            print(display)
        return print_out

    def save(self, name, bestInd, scores, g, t, last, out):
        a = os.path.join(os.path.join(self.path, 'results.txt'))
        with open(a, "w") as f:
            try:
                method = self.pipeline.steps[-1][1].__class__.__name__
            except Exception:
                method = str(self.pipeline)
            bestSubset = [self.cols[i] for i in range(len(self.cols)) if bestInd[i]]
            score_train, y_true, y_pred = fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=bestInd,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=0,
                cv=self.cv
            )
            if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
                tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
                string_tmp = f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}" + os.linesep
            elif isinstance(self.pipeline.steps[-1][1], RegressorMixin):
                string_tmp = f"Regression residuals (first 5): {(y_true - y_pred).head().tolist()}" + os.linesep
            string = (
                    f"Filter: {name}" + os.linesep +
                    f"Iterations: {self.Gmax}" + os.linesep +
                    f"Iterations Performed: {g}" + os.linesep +
                    f"Latest Improvement: {last}" + os.linesep +
                    f"Latest Improvement (Ratio): {1 - (last / g)}" + os.linesep +
                    f"Cross-validation strategy: {str(self.cv)}" + os.linesep +
                    f"Method: {method}" + os.linesep +
                    f"Best Score: {score_train}" + os.linesep +
                    string_tmp +
                    f"Best Subset: {bestSubset}" + os.linesep +
                    f"Number of Features: {len(bestSubset)}" + os.linesep +
                    f"Number of Features (Ratio): {len(bestSubset) / len(self.cols)}" + os.linesep +
                    f"Score of Features: {scores}" + os.linesep +
                    f"Execution Time: {round(t.total_seconds())} ({t})" + os.linesep +
                    f"Memory: {psutil.virtual_memory()}"
            )
            f.write(string)
        a = os.path.join(self.path, 'log.txt')
        with open(a, "a") as f:
            f.write(out)
        # Pipeline saving
        pipeline_path = os.path.join(self.path, 'pipeline.joblib')
        joblib.dump(self.pipeline, pipeline_path)


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
        # --- Extraction / cast numpy
        X = df.drop(columns=[target])
        y = df[target].to_numpy()
        # S'assure que X est numérique (si besoin, adapter en amont)
        X = X.astype(float)
        n, p = X.shape
        cols = X.columns.to_list()
        # --- Standardisation colonne par colonne (ddof=1, comme pandas.corr)
        # Xc: centrée ; std: écart-type échantillon ; Z: variance unitaire
        Xv = X.to_numpy(copy=False)
        Xc = Xv - np.nanmean(Xv, axis=0, keepdims=True)
        # Remplace NaN par 0 après centrage (optionnel selon tes données)
        Xc = np.nan_to_num(Xc, copy=False)
        # std échantillon (ddof=1) ; évite division par 0
        stdX = Xc.std(axis=0, ddof=1)
        stdX[stdX == 0] = 1.0
        Z = Xc / stdX  # (n, p)
        # --- Relevance ∈ [0,1]
        if is_quantitative:
            # Corrélation vectorisée r(Xj, y) puis min-max → [0,1]
            yc = y.astype(float) - float(np.mean(y))
            # Si variance(y)==0
            denom_y = np.sqrt(np.sum((yc ** 2)))
            if denom_y == 0:
                r = np.zeros(p, dtype=float)
            else:
                # corr(Xj, y) = (Xc_j · yc) / (||Xc_j|| * ||yc||)
                # comme Z_j = Xc_j / stdX_j, et stdX_j^2 = sum(Xc_j^2)/(n-1),
                # alors corr = (Z^T @ (yc/std_y)) / (n-1)
                # On reste sur une forme simple et stable :
                r = (Xc.T @ yc) / (stdX * denom_y)  # (p,)
            r = np.nan_to_num(r, nan=0.0)
            # min-max scaling
            r_min, r_max = float(np.min(r)), float(np.max(r))
            if r_max > r_min:
                relevance = (r - r_min) / (r_max - r_min)
            else:
                relevance = np.zeros_like(r)
        else:
            # f_classif déjà vectorisé
            f_values, _ = f_classif(Xv, y)
            f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
            f_min, f_max = float(np.min(f_values)), float(np.max(f_values))
            if f_max > f_min:
                relevance = (f_values - f_min) / (f_max - f_min)
            else:
                relevance = np.zeros_like(f_values)
        # --- Sélection gloutonne avec redondance incrémentale
        selected = np.zeros(p, dtype=bool)
        red_sum = np.zeros(p, dtype=float)  # somme des redondances (transformées en [0,1]) accumulées
        S = []
        mrmr_scores_list = {}
        # 1) Première feature = max relevance
        j0 = int(np.argmax(relevance))
        S.append(cols[j0])
        selected[j0] = True
        mrmr_scores_list[cols[j0]] = float(relevance[j0])
        # 2) Boucle : à chaque étape, on met à jour la redondance via un produit Z^T @ z_s
        #    puis on choisit la feature maximisant relevance - (red_sum / t)
        #    NB: corr en [-1,1] → [0,1] via (r+1)/2, cumulée dans red_sum
        ZT = Z.T  # (p, n) pour éviter des transpositions répétées
        for _t in range(1, p):  # on produit l'ordre complet (tu as demandé d'ignorer l'arrêt à k)
            # Dernière sélection s
            s = int(np.where(selected)[0][-1])
            # Corrélation de toutes les variables avec Z[:, s] (coût O(n p))
            z_s = Z[:, s]  # (n,)
            corr_vec = (ZT @ z_s) / (n - 1.0)  # (p,) corrélation de Pearson (ddof=1)
            # Transforme vers [0,1] et cumule
            red_sum += (np.clip(corr_vec, -1.0, 1.0) + 1.0) * 0.5
            # Score mRMR courant pour les candidates
            t_cur = np.count_nonzero(selected)
            scores = relevance - (red_sum / t_cur)
            # Invalide les déjà sélectionnées
            scores[selected] = -np.inf
            # Prochaine feature
            j_next = int(np.argmax(scores))
            S.append(cols[j_next])
            selected[j_next] = True
            mrmr_scores_list[cols[j_next]] = float(scores[j_next])
            # (Optionnel) petit early-exit si tout est sélectionné
            if t_cur + 1 == p:
                break
        return S, mrmr_scores_list

    @staticmethod
    def surf_selection(df, target, is_quantitative):
        X_df = df.drop(columns=[target])
        y = df[target].to_numpy()
        feature_names = X_df.columns.tolist()
        X = MinMaxScaler().fit_transform(X_df.to_numpy(dtype=float))
        n_samples, n_features = X.shape
        D = pairwise_distances(X, metric="euclidean", n_jobs=1)
        if n_samples > 1:
            T = D[np.triu_indices(n_samples, 1)].mean()
        else:
            T = 0.0
        nbrs = NearestNeighbors(radius=T, algorithm="auto", metric="euclidean").fit(X)
        W = np.zeros(n_features, dtype=float)
        if not is_quantitative:
            classes, counts = np.unique(y, return_counts=True)
            priors = {c: counts[i] / n_samples for i, c in enumerate(classes)}
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                neigh_idx = nbrs.radius_neighbors(xi.reshape(1, -1), return_distance=False)[0]
                neigh_idx = neigh_idx[neigh_idx != i]
                if neigh_idx.size == 0:
                    continue
                hits = neigh_idx[y[neigh_idx] == yi]
                if hits.size > 0:
                    W -= np.mean(np.abs(X[hits] - xi), axis=0)
                denom = max(1e-12, (1.0 - priors.get(yi, 0.0)))
                miss_acc = np.zeros(n_features, dtype=float)
                for c in classes:
                    if c == yi:
                        continue
                    miss_c = neigh_idx[y[neigh_idx] == c]
                    if miss_c.size == 0:
                        continue
                    w_c = priors[c] / denom
                    miss_acc += w_c * np.mean(np.abs(X[miss_c] - xi), axis=0)
                W += miss_acc
        else:
            y = y.astype(float)
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                neigh_idx = nbrs.radius_neighbors(xi.reshape(1, -1), return_distance=False)[0]
                neigh_idx = neigh_idx[neigh_idx != i]
                if neigh_idx.size == 0:
                    continue
                dy = np.abs(y[neigh_idx] - yi)
                s = dy.sum()
                if s <= 0:
                    continue
                w = dy / s
                diffs = np.abs(X[neigh_idx] - xi)
                W += (w[:, None] * diffs).sum(axis=0)
        if n_samples > 0:
            W /= n_samples
        order = np.argsort(-W)
        features_sorted = [feature_names[k] for k in order]
        scores_sorted = W[order]
        return features_sorted, scores_sorted

    def start(self, pid):
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        score, col, vector, s = -np.inf, [], [], -np.inf
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
            s = fitness(train=self.train, test=self.test, columns=self.cols, ind=v, target=self.target,
                        pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]
            if s > score:
                same = 0
                score, vector = s, v
                col = [self.cols[i] for i in range(len(self.cols)) if v[i]]
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            print_out = self.print_(print_out=print_out, name=self.v_name, pid=pid, maxi=score, best=s, mean=s,
                                    feats=len(col), time_exe=time_instant, time_total=time_debut, g=G, cpt=same,
                                    verbose=self.verbose) + "\n"
            if time.time() - debut >= self.Tmax or G == len(num_features):
                stop = True
            if G % 10 == 0 or G == self.Gmax or stop:
                self.save(name=self.method, bestInd=vector, scores=filter_scores, g=G,
                          t=timedelta(seconds=(time.time() - debut)), last=G - same, out=print_out)
                print_out = ""
                if stop:
                    break
        return score, vector, col, self.pipeline, pid, self.v_name, G - same, G
