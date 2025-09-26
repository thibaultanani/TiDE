"""Filter-based feature selection strategies."""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from feature_selections import FeatureSelection
from utility import create_directory, fitness


ScoreListing = Sequence[Tuple[str, float]] | Sequence[float]


class Filter(FeatureSelection):
    """Parent implementation shared by the available filter methods.

    Parameters specific to this strategy
    -----------------------------------
    method: str | None
        Name of the ranking heuristic to apply. Accepted values are
        ``'correlation'``, ``'anova'``, ``'mutual information'``, ``'mrmr'`` and
        ``'surf'``.
    """

    def __init__(
        self,
        name,
        target,
        pipeline,
        train,
        test=None,
        cv=None,
        drops=None,
        scoring=None,
        Gmax=None,
        Tmax=None,
        ratio=None,
        suffix=None,
        verbose=None,
        output=None,
        method: str | None = None,
        warm_start: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name,
            target,
            pipeline,
            train,
            test,
            cv,
            drops,
            scoring,
            Gmax,
            Tmax,
            ratio,
            suffix,
            verbose,
            output,
            warm_start=warm_start,
            seed=seed,
        )

        self._estimator = self._resolve_estimator()
        self.quantitative = not isinstance(self._estimator, ClassifierMixin)

        self.method, self.func, self.v_name = self._resolve_strategy(method)
        self.path = Path(self.path) / f"{self.method.lower()}{self.suffix}"
        create_directory(path=self.path)

    def _resolve_estimator(self) -> BaseEstimator:
        """Return the final estimator from the provided pipeline."""

        if hasattr(self.pipeline, "steps") and self.pipeline.steps:
            return self.pipeline.steps[-1][1]
        return self.pipeline

    def _resolve_strategy(self, method: str | None) -> tuple[str, Callable[[pd.DataFrame, str], tuple[List[str], ScoreListing]], str]:
        """Return the filter strategy associated with ``method``."""

        method_key = (method or "Correlation").strip().lower()

        def correlation(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
            return self.correlation_selection(df=df, target=target)

        def anova(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
            return self.anova_selection(df=df, target=target)

        def mutual_info(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
            return self.mutual_info_selection(df=df, target=target)

        def mrmr(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
            return self.mrmr_selection(df=df, target=target, is_quantitative=self.quantitative)

        def surf(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
            return self.surf_selection(df=df, target=target, is_quantitative=self.quantitative)

        strategies = {
            "correlation": ("Correlation", correlation, "CORR"),
            "anova": ("Anova", anova, "ANOV"),
            "mutual information": ("Mutual Information", mutual_info, "MI  "),
            "mrmr": ("MRMR", mrmr, "MRMR"),
            "surf": ("SURF", surf, "SURF"),
        }

        if method_key not in strategies:
            valid = "', '".join(sorted(strategies))
            raise ValueError(f"Unknown filter method '{method}'. Available methods: '{valid}'.")
        return strategies[method_key]

    @staticmethod
    def print_(
        print_out: str,
        name: str,
        pid: int,
        maxi: float,
        best: float,
        mean: float,
        feats: int,
        time_exe: timedelta,
        time_total: timedelta,
        g: int,
        cpt: int,
        verbose: bool,
    ) -> str:
        """Pretty-print the optimisation status for filter methods."""

        display = (
            "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}"
            "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}"
        ).format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt)
        if verbose:
            print(display)
        return print_out + display

    def save(
        self,
        name: str,
        bestInd: Sequence[bool],
        bestTime: timedelta,
        scores: ScoreListing,
        g: int,
        t: timedelta,
        last: int,
        out: str,
    ) -> None:
        """Persist the filter summary and fitted pipeline to disk."""

        results_path = Path(self.path) / "results.txt"
        with results_path.open("w", encoding="utf-8") as f:
            try:
                method = self._estimator.__class__.__name__
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
                cv=self.cv,
                rng=self._rng,
            )
            if isinstance(self._estimator, ClassifierMixin):
                tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
                string_tmp = f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}" + os.linesep
            elif isinstance(self._estimator, RegressorMixin):
                string_tmp = f"Regression residuals (first 5): {(y_true - y_pred).head().tolist()}" + os.linesep
            else:
                string_tmp = ""
            string = (
                f"Filter: {name}"
                + os.linesep
                + f"Iterations: {self.Gmax}"
                + os.linesep
                + f"Iterations Performed: {g}"
                + os.linesep
                + f"Latest Improvement: {last}"
                + os.linesep
                + f"Latest Improvement (Ratio): {1 - (last / g if g else 0)}"
                + os.linesep
                + f"Latest Improvement (Time): {round(bestTime.total_seconds())} ({bestTime})"
                + os.linesep
                + f"Cross-validation strategy: {str(self.cv)}"
                + os.linesep
                + f"Method: {method}"
                + os.linesep
                + f"Best Score: {score_train}"
                + os.linesep
                + string_tmp
                + f"Best Subset: {bestSubset}"
                + os.linesep
                + f"Number of Features: {len(bestSubset)}"
                + os.linesep
                + f"Number of Features (Ratio): {len(bestSubset) / len(self.cols)}"
                + os.linesep
                + f"Score of Features: {scores}"
                + os.linesep
                + f"Execution Time: {round(t.total_seconds())} ({t})"
                + os.linesep
                + f"Memory: {psutil.virtual_memory()}"
            )
            f.write(string)

        log_path = Path(self.path) / "log.txt"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(out)

        pipeline_path = Path(self.path) / "pipeline.joblib"
        joblib.dump(self.pipeline, pipeline_path)

    @staticmethod
    def correlation_selection(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
        """Rank features by their Pearson correlation with ``target``."""

        corr_results = []
        for column in df.columns:
            if column == target:
                continue
            coef, _ = pearsonr(df[column], df[target].values)
            corr_results.append((column, float(coef)))
        corr_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in corr_results]
        return sorted_features, corr_results

    @staticmethod
    def anova_selection(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
        """Rank features using an ANOVA F-test."""

        X = df.drop([target], axis=1)
        y = df[target]
        f_values, _ = f_classif(X, y)
        f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
        f_results = list(zip(X.columns.tolist(), f_values))
        f_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in f_results]
        return sorted_features, f_results

    @staticmethod
    def mutual_info_selection(df: pd.DataFrame, target: str) -> tuple[List[str], ScoreListing]:
        """Rank features using mutual information with the target."""

        X = df.drop([target], axis=1)
        y = df[target]
        mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        mi_results = list(zip(X.columns.tolist(), mi_scores))
        mi_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in mi_results]
        return sorted_features, mi_results

    @staticmethod
    def mrmr_selection(
        df: pd.DataFrame,
        target: str,
        is_quantitative: bool = False,
    ) -> tuple[List[str], ScoreListing]:
        """Rank features using the minimum redundancy maximum relevance criterion."""

        X = df.drop(columns=[target]).astype(float)
        y = df[target].to_numpy()
        n, p = X.shape
        cols = X.columns.to_list()
        Xv = X.to_numpy(copy=False)
        Xc = Xv - np.nanmean(Xv, axis=0, keepdims=True)
        Xc = np.nan_to_num(Xc, copy=False)
        stdX = Xc.std(axis=0, ddof=1)
        stdX[stdX == 0] = 1.0
        Z = Xc / stdX
        if is_quantitative:
            yc = y.astype(float) - float(np.mean(y))
            denom_y = np.sqrt(np.sum((yc**2)))
            if denom_y == 0:
                r = np.zeros(p, dtype=float)
            else:
                r = (Xc.T @ yc) / (stdX * denom_y)
            r = np.nan_to_num(r, nan=0.0)
            r_min, r_max = float(np.min(r)), float(np.max(r))
            relevance = (r - r_min) / (r_max - r_min) if r_max > r_min else np.zeros_like(r)
        else:
            f_values, _ = f_classif(Xv, y)
            f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
            f_min, f_max = float(np.min(f_values)), float(np.max(f_values))
            relevance = (f_values - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(f_values)

        selected = np.zeros(p, dtype=bool)
        red_sum = np.zeros(p, dtype=float)
        order = []
        scores_dict: dict[str, float] = {}
        ZT = Z.T

        j0 = int(np.argmax(relevance))
        order.append(cols[j0])
        selected[j0] = True
        scores_dict[cols[j0]] = float(relevance[j0])

        for _ in range(1, p):
            s = int(np.where(selected)[0][-1])
            z_s = Z[:, s]
            corr_vec = (ZT @ z_s) / (n - 1.0)
            red_sum += (np.clip(corr_vec, -1.0, 1.0) + 1.0) * 0.5
            t_cur = np.count_nonzero(selected)
            scores = relevance - (red_sum / t_cur)
            scores[selected] = -np.inf
            j_next = int(np.argmax(scores))
            order.append(cols[j_next])
            selected[j_next] = True
            scores_dict[cols[j_next]] = float(scores[j_next])
            if t_cur + 1 == p:
                break
        return order, [(feat, scores_dict[feat]) for feat in order]

    @staticmethod
    def surf_selection(
        df: pd.DataFrame,
        target: str,
        is_quantitative: bool,
    ) -> tuple[List[str], ScoreListing]:
        """Rank features using the SURF (Spatial Uniform ReliefF) algorithm."""

        X_df = df.drop(columns=[target])
        y = df[target].to_numpy()
        feature_names = X_df.columns.tolist()
        X = MinMaxScaler().fit_transform(X_df.to_numpy(dtype=float))
        n_samples, n_features = X.shape
        D = pairwise_distances(X, metric="euclidean", n_jobs=1)
        T = D[np.triu_indices(n_samples, 1)].mean() if n_samples > 1 else 0.0
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
        return features_sorted, list(zip(features_sorted, scores_sorted))

    def start(self, pid: int):
        """Execute the selected filter and evaluate incremental subsets."""

        debut = time.time()
        create_directory(path=self.path)
        print_out = ""
        self.reset_rng()
        score, col, vector, best_time = -np.inf, [], [], timedelta(seconds=0)
        same = 0

        sorted_features, filter_scores = self.func(self.train, self.target)
        percentiles = [val for val in range(1, 101)]
        num_features = [int(round(len(sorted_features) * (val / 100.0))) for val in percentiles]
        num_features = [val for val in num_features if val >= 1]
        num_features = list(dict.fromkeys(num_features))

        generation = 0
        for generation, feature_count in enumerate(num_features, start=1):
            instant = time.time()
            top_k_features = sorted_features[:feature_count]
            candidate = [0] * self.D
            for var in top_k_features:
                candidate[self.cols.get_loc(var)] = 1
            s = fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=candidate,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio,
                cv=self.cv,
                rng=self._rng,
            )[0]
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            if s > score:
                same = 0
                score = s
                vector = candidate
                best_time = time_debut
                col = [self.cols[i] for i in range(len(self.cols)) if vector[i]]
            else:
                same += 1
            print_out = self.print_(
                print_out=print_out,
                name=self.v_name,
                pid=pid,
                maxi=score,
                best=s,
                mean=s,
                feats=len(col),
                time_exe=time_instant,
                time_total=time_debut,
                g=generation,
                cpt=same,
                verbose=self.verbose,
            ) + "\n"
            stop = (time.time() - debut) >= self.Tmax or generation == len(num_features) or generation == self.Gmax
            if generation % 10 == 0 or generation == self.Gmax or stop:
                self.save(
                    name=self.method,
                    bestInd=vector,
                    bestTime=best_time,
                    scores=filter_scores,
                    g=generation,
                    t=timedelta(seconds=(time.time() - debut)),
                    last=generation - same,
                    out=print_out,
                )
                print_out = ""
                if stop:
                    break

        return score, vector, col, best_time, self.pipeline, pid, self.v_name, generation - same, generation
