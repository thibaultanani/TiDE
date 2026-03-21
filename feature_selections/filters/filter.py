"""Filter-based feature selection strategies."""

from __future__ import annotations

import json
import os
import time
from functools import partial
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import check_cv
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from feature_selections import FeatureSelection
from helper import create_directory, fitness


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
        selection_mode: str | None = None,
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
        self.selection_mode = self._resolve_selection_mode(selection_mode)
        self.path = Path(self.path) / f"{self.method.lower()}{self.suffix}"
        create_directory(path=self.path)
        self._best_history: list[tuple[float, float, int]] = []

    def reset_tracking(self) -> None:
        """Clear the time-to-best tracking history."""

        self._best_history = []

    def seed_full_subset_tracking(self) -> float:
        """Seed the time-to-best curve with the score of the full feature set."""

        full_subset = np.ones(self.D, dtype=bool)
        full_score = float(
            fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=full_subset,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio,
                cv=self.cv,
                rng=self._rng,
            )[0]
        )
        self.track_best(full_score, timedelta(seconds=0), self.D)
        return full_score

    def track_best(self, score: float, time_total: timedelta, n_features: int) -> None:
        """Record a new best score with its elapsed time."""

        if not np.isfinite(score):
            return
        elapsed = float(time_total.total_seconds())
        if not self._best_history:
            self._best_history.append((elapsed, float(score), int(n_features)))
            return
        if score > self._best_history[-1][1]:
            self._best_history.append((elapsed, float(score), int(n_features)))

    def _write_best_points(self) -> tuple[Path, float, list[dict[str, float | int]]]:
        """Write the normalised time-to-best curve and return its AUAC."""

        points_path = Path(self.path) / "time_to_best_curve.json"
        curve_points = self.build_time_to_best_curve(self._best_history)
        auac = self.compute_auac(curve_points)
        with points_path.open("w", encoding="utf-8") as f:
            json.dump(curve_points, f, ensure_ascii=True)
        return points_path, auac, curve_points

    def _resolve_estimator(self) -> BaseEstimator:
        """Return the final estimator from the provided pipeline."""

        if hasattr(self.pipeline, "steps") and self.pipeline.steps:
            return self.pipeline.steps[-1][1]
        return self.pipeline

    def _internal_evaluation_mode(self) -> str:
        """Return the internal evaluation regime used by the selector."""

        return "cv" if self.cv is not None else "holdout"

    def _resolve_strategy(self, method: str | None) -> tuple[str, Callable[[pd.DataFrame, str], tuple[List[str], ScoreListing]], str]:
        """Return the filter strategy associated with ``method``."""

        method_key = (method or "Correlation").strip().lower()

        strategies = {
            "correlation": ("Correlation", Filter.correlation_selection, "CORR"),
            "anova": ("Anova", Filter.anova_selection, "ANOV"),
            "mutual information": (
                "Mutual Information",
                partial(Filter.mutual_info_selection, is_quantitative=self.quantitative),
                "MI  ",
            ),
            "mrmr": ("MRMR", partial(Filter.mrmr_selection, is_quantitative=self.quantitative), "MRMR"),
            "surf": ("SURF", partial(Filter.surf_selection, is_quantitative=self.quantitative), "SURF"),
        }

        if method_key not in strategies:
            valid = "', '".join(sorted(strategies))
            raise ValueError(f"Unknown filter method '{method}'. Available methods: '{valid}'.")
        return strategies[method_key]

    def _resolve_selection_mode(self, selection_mode: str | None) -> str:
        """Return the filter selection protocol to use."""

        if selection_mode is None:
            return "cv_majority" if self.cv is not None else "global_ranking"
        mode = selection_mode.strip().lower()
        valid = {"global_ranking", "cv_majority"}
        if mode not in valid:
            raise ValueError(f"Unknown selection_mode '{selection_mode}'. Available modes: {sorted(valid)}.")
        if mode == "cv_majority" and self.cv is None:
            raise ValueError("selection_mode='cv_majority' requires a cross-validation strategy via 'cv'.")
        return mode

    @staticmethod
    def _mode_label(selection_mode: str) -> str:
        """Return a human-readable label for ``selection_mode``."""

        labels = {
            "global_ranking": "global ranking",
            "cv_majority": "per-fold ranking + majority vote",
        }
        return labels[selection_mode]

    def _indicator_from_features(self, features: Sequence[str]) -> list[int]:
        """Return a binary indicator matching ``features``."""

        candidate = [0] * self.D
        for var in features:
            candidate[self.cols.get_loc(var)] = 1
        return candidate

    def _score_candidate(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, candidate: Sequence[bool]) -> float:
        """Evaluate ``candidate`` on an explicit train/validation split."""

        return float(
            fitness(
                train=train_df,
                test=valid_df,
                columns=self.cols,
                ind=candidate,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio,
                cv=None,
                rng=self._rng,
            )[0]
        )

    def _score_internal_candidate(self, candidate: Sequence[bool], *, ratio: float | None = None) -> float:
        """Evaluate ``candidate`` with the method's internal selection protocol."""

        return float(
            fitness(
                train=self.train,
                test=None if self.cv is not None else self.test,
                columns=self.cols,
                ind=candidate,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio if ratio is None else ratio,
                cv=self.cv,
                rng=self._rng,
            )[0]
        )

    def _majority_vote_subset(
        self,
        fold_summaries: Sequence[dict[str, object]],
        majority_threshold: int,
    ) -> list[str]:
        """Return the current consensus subset built from per-fold best subsets."""

        feature_votes: dict[str, int] = {}
        for fold_summary in fold_summaries:
            for feature in fold_summary["best_subset"]:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1

        subset = [feature for feature in self.cols if feature_votes.get(feature, 0) >= majority_threshold]
        if not subset and feature_votes:
            max_votes = max(feature_votes.values())
            subset = [feature for feature in self.cols if feature_votes.get(feature, 0) == max_votes]
        if not subset and fold_summaries:
            best_fold = max(fold_summaries, key=lambda item: float(item["best_score"]))
            subset = list(best_fold["best_subset"])
        return subset

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

        results_json_path = Path(self.path) / "results.json"
        results_path = Path(self.path) / "results.txt"
        try:
            method = self._estimator.__class__.__name__
        except Exception:
            method = str(self.pipeline)
        bestSubset = [self.cols[i] for i in range(len(self.cols)) if bestInd[i]]
        score_train, y_true, y_pred, fold_scores, fold_std = fitness(
            train=self.train,
            test=None if self.cv is not None else self.test,
            columns=self.cols,
            ind=bestInd,
            target=self.target,
            pipeline=self.pipeline,
            scoring=self.scoring,
            ratio=0,
            cv=self.cv,
            rng=self._rng,
        )
        metrics_payload: dict[str, int | list[float]] = {}
        if isinstance(self._estimator, ClassifierMixin):
            tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
            string_tmp = f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}" + os.linesep
            metrics_payload = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        elif isinstance(self._estimator, RegressorMixin):
            residuals = (y_true - y_pred).head().tolist()
            string_tmp = f"Regression residuals (first 5): {residuals}" + os.linesep
            metrics_payload = {"residuals_first_5": residuals}
        else:
            string_tmp = ""
        if fold_scores is None:
            cv_string = "CV Fold Scores: None" + os.linesep + "CV Fold Std: None" + os.linesep
        else:
            cv_string = (
                f"CV Fold Scores: {fold_scores}" + os.linesep + f"CV Fold Std: {fold_std}" + os.linesep
            )
        points_path, auac, curve_points = self._write_best_points()
        results_payload = {
            "strategy_type": "filter",
            "name": name,
            "method_mode": self.selection_mode,
            "method_mode_label": self._mode_label(self.selection_mode),
            "internal_evaluation_mode": self._internal_evaluation_mode(),
            "sparsity_ratio": float(self.ratio),
            "time_budget_seconds": float(self.Tmax),
            "selection_mode": self.selection_mode,
            "selection_mode_label": self._mode_label(self.selection_mode),
            "iterations": int(self.Gmax),
            "iterations_performed": int(g),
            "latest_improvement": int(last),
            "latest_improvement_ratio": float(1 - (last / g if g else 0)),
            "latest_improvement_time_seconds": float(bestTime.total_seconds()),
            "cross_validation_strategy": str(self.cv),
            "method": method,
            "ba_cv": float(score_train),
            "best_subset": bestSubset,
            "number_of_features": len(bestSubset),
            "number_of_features_ratio": float(len(bestSubset) / len(self.cols)),
            "feature_scores": list(scores),
            "execution_time_seconds": float(t.total_seconds()),
            "auac_time_normalized": float(auac),
            "time_to_best_points_file": points_path.name,
            "time_to_best_points": curve_points,
            "metrics": metrics_payload,
            "cv_fold_scores": None if fold_scores is None else [float(score) for score in fold_scores],
            "cv_fold_std": None if fold_std is None else float(fold_std),
            "memory": str(psutil.virtual_memory()),
        }
        with results_json_path.open("w", encoding="utf-8") as f:
            json.dump(results_payload, f, ensure_ascii=True, indent=2)
        with results_path.open("w", encoding="utf-8") as f:
            string = (
                f"Filter: {name}"
                + os.linesep
                + f"Method Mode: {self.selection_mode} ({self._mode_label(self.selection_mode)})"
                + os.linesep
                + f"Internal Evaluation Mode: {self._internal_evaluation_mode()}"
                + os.linesep
                + f"Sparsity Ratio: {self.ratio}"
                + os.linesep
                + f"Time Budget (s): {self.Tmax}"
                + os.linesep
                + f"Iterations: {self.Gmax}"
                + os.linesep
                + f"Iterations Performed: {g}"
                + os.linesep
                + f"Selection Mode: {self.selection_mode} ({self._mode_label(self.selection_mode)})"
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
                + f"BA CV: {score_train}"
                + os.linesep
                + string_tmp
                + cv_string
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
                + f"AUAC (time-normalized): {auac:.6f}"
                + os.linesep
                + f"Time-to-best points: {points_path.name}"
                + os.linesep
                + f"Memory: {psutil.virtual_memory()}"
            )
            f.write(string)

        log_path = Path(self.path) / "log.txt"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(out)
            if out and not out.endswith(os.linesep):
                f.write(os.linesep)
            f.write(f"Sparsity Ratio: {self.ratio}" + os.linesep)
            f.write(f"AUAC (time-normalized): {auac:.6f}" + os.linesep)
            f.write(f"Time-to-best points: {json.dumps(curve_points, ensure_ascii=True)}" + os.linesep)

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
            coef = float(np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0))
            corr_results.append((column, abs(coef)))
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
    def mutual_info_selection(
        df: pd.DataFrame,
        target: str,
        is_quantitative: bool,
    ) -> tuple[List[str], ScoreListing]:
        """Rank features using mutual information with the target."""

        X = df.drop([target], axis=1)
        y = df[target]
        if is_quantitative:
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
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
            r = np.abs(np.nan_to_num(r, nan=0.0))
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

        j = int(np.argmax(relevance))
        order.append(cols[j])
        selected[j] = True
        scores_dict[cols[j]] = float(relevance[j])

        corr_vec = np.clip(np.abs((ZT @ Z[:, j]) / (n - 1.0)), 0.0, 1.0)
        red_sum += corr_vec

        for _ in range(1, p):
            t_cur = int(np.count_nonzero(selected))
            t_cur = max(1, t_cur)

            scores = relevance - (red_sum / t_cur)
            scores[selected] = -np.inf

            j_next = int(np.argmax(scores))
            if not np.isfinite(scores[j_next]):
                break

            order.append(cols[j_next])
            selected[j_next] = True
            scores_dict[cols[j_next]] = float(scores[j_next])

            corr_vec = np.clip(np.abs((ZT @ Z[:, j_next]) / (n - 1.0)), 0.0, 1.0)
            red_sum += corr_vec

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

    def _start_global_ranking(self, pid: int):
        """Execute the legacy global-ranking filter protocol."""

        debut = time.time()
        print_out = ""
        score, col, vector, best_time = -np.inf, [], [], timedelta(seconds=0)
        same = 0

        sorted_features, filter_scores = self.func(self.train, self.target)
        num_features = list(range(1, len(sorted_features) + 1))

        generation = 0
        for generation, feature_count in enumerate(num_features, start=1):
            instant = time.time()
            top_k_features = sorted_features[:feature_count]
            candidate = self._indicator_from_features(top_k_features)
            s = self._score_internal_candidate(candidate)
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            if s > score:
                same = 0
                score = s
                vector = candidate
                best_time = time_debut
                col = [self.cols[i] for i in range(len(self.cols)) if vector[i]]
                self.track_best(score, time_debut, len(col))
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

    def _start_cv_majority(self, pid: int):
        """Execute the per-fold ranking + majority-vote filter protocol."""

        debut = time.time()
        print_out = ""
        global_best_score = float(self._best_history[0][1]) if self._best_history else float("-inf")
        global_best_vector = self._indicator_from_features(self.cols) if self._best_history else [0] * self.D
        global_best_col: list[str] = list(self.cols) if self._best_history else []
        best_time = timedelta(seconds=0)
        same = 0
        generation = 0
        current_consensus_score = global_best_score
        current_consensus_subset = list(global_best_col)

        X = self.train.drop(columns=[self.target])
        y = self.train[self.target]
        splitter = check_cv(cv=self.cv, y=y, classifier=isinstance(self._estimator, ClassifierMixin))
        splits = list(splitter.split(X, y))
        majority_threshold = max(1, (len(splits) // 2) + 1)
        fold_summaries: list[dict[str, object]] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
            if (time.time() - debut) >= self.Tmax or generation >= self.Gmax:
                break
            fold_train = self.train.iloc[train_idx].reset_index(drop=True)
            fold_valid = self.train.iloc[valid_idx].reset_index(drop=True)
            sorted_features, fold_scores = self.func(fold_train, self.target)
            fold_best_score = float("-inf")
            fold_best_features: list[str] = []
            fold_summary: dict[str, object] = {
                "fold": fold_idx,
                "ranking_scores": list(fold_scores),
                "best_score": float("-inf"),
                "best_subset": [],
            }
            fold_summaries.append(fold_summary)

            for feature_count in range(1, len(sorted_features) + 1):
                if (time.time() - debut) >= self.Tmax or generation >= self.Gmax:
                    break
                instant = time.time()
                candidate_features = sorted_features[:feature_count]
                candidate = self._indicator_from_features(candidate_features)
                score = self._score_candidate(fold_train, fold_valid, candidate)
                generation += 1
                time_instant = timedelta(seconds=(time.time() - instant))
                time_total = timedelta(seconds=(time.time() - debut))
                consensus_updated = False
                if score > fold_best_score:
                    fold_best_score = score
                    fold_best_features = list(candidate_features)
                    fold_summary["best_score"] = float(fold_best_score)
                    fold_summary["best_subset"] = fold_best_features
                    current_consensus_subset = self._majority_vote_subset(fold_summaries, majority_threshold)
                    current_consensus_vector = self._indicator_from_features(current_consensus_subset)
                    current_consensus_score = self._score_internal_candidate(current_consensus_vector)
                    consensus_updated = True
                    if current_consensus_score > global_best_score:
                        same = 0
                        global_best_score = current_consensus_score
                        global_best_vector = current_consensus_vector
                        global_best_col = list(current_consensus_subset)
                        best_time = time_total
                        self.track_best(global_best_score, time_total, len(global_best_col))
                    else:
                        same += 1
                else:
                    same += 1
                print_out = self.print_(
                    print_out=print_out,
                    name=self.v_name,
                    pid=pid,
                    maxi=global_best_score,
                    best=current_consensus_score if consensus_updated else global_best_score,
                    mean=score,
                    feats=len(current_consensus_subset if consensus_updated else global_best_col),
                    time_exe=time_instant,
                    time_total=time_total,
                    g=generation,
                    cpt=same,
                    verbose=self.verbose,
                ) + "\n"
                stop = (time.time() - debut) >= self.Tmax or generation == self.Gmax
                if generation % 10 == 0 or generation == self.Gmax or stop:
                    self.save(
                        name=self.method,
                        bestInd=global_best_vector,
                        bestTime=best_time,
                        scores=fold_summaries,
                        g=generation,
                        t=timedelta(seconds=(time.time() - debut)),
                        last=generation - same,
                        out=print_out,
                    )
                    print_out = ""
                    if stop:
                        break

        final_subset = self._majority_vote_subset(fold_summaries, majority_threshold)

        final_vector = self._indicator_from_features(final_subset)
        final_score = self._score_internal_candidate(final_vector)
        final_time = timedelta(seconds=(time.time() - debut))
        if final_score > global_best_score:
            global_best_score = float(final_score)
            global_best_vector = final_vector
            global_best_col = final_subset
            best_time = final_time
            self.track_best(global_best_score, final_time, len(global_best_col))

        self.save(
            name=self.method,
            bestInd=global_best_vector,
            bestTime=best_time,
            scores=fold_summaries,
            g=generation,
            t=timedelta(seconds=(time.time() - debut)),
            last=generation - same,
            out=print_out,
        )
        return (
            global_best_score,
            global_best_vector,
            global_best_col,
            best_time,
            self.pipeline,
            pid,
            self.v_name,
            generation - same,
            generation,
        )

    def start(self, pid: int):
        """Execute the selected filter protocol."""

        create_directory(path=self.path)
        self.reset_rng()
        self.reset_tracking()
        self.seed_full_subset_tracking()
        if self.selection_mode == "cv_majority":
            return self._start_cv_majority(pid)
        return self._start_global_ranking(pid)
