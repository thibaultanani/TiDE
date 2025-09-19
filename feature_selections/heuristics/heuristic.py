from __future__ import annotations

import abc
import os
from pathlib import Path
from datetime import timedelta
from typing import Optional, Sequence

import joblib
import psutil
from sklearn.base import ClassifierMixin, RegressorMixin

from feature_selections import FeatureSelection
from utility import fitness


class Heuristic(FeatureSelection):
    """Parent class for population-based and single-solution heuristics."""

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
        N: Optional[int] = None,
        Gmax=None,
        Tmax=None,
        ratio=None,
        suffix=None,
        verbose=None,
        output=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, Gmax, Tmax, ratio, suffix, verbose, output)
        self.N = N if N is not None else 100

    @abc.abstractmethod
    def start(self, pid: int) -> None:
        """Execute the heuristic optimisation process."""

    @staticmethod
    def pprint_(
        print_out: str,
        name: str,
        pid: int,
        maxi: float,
        best: float,
        mean: float,
        feats: int,
        time_exe: timedelta,
        time_total: timedelta,
        entropy: float,
        g: int,
        cpt: int,
        verbose: bool,
    ) -> str:
        """Pretty-print the optimisation status and store it in ``print_out``."""

        display = (
            "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}"
            "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}    entropy : {:2.3f}"
        ).format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt, entropy)
        print_out = print_out + display
        if verbose:
            print(display)
        return print_out

    @staticmethod
    def sprint_(
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
        """Pretty-print optimisation status without the entropy information."""

        display = (
            "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}"
            "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}"
        ).format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt)
        print_out = print_out + display
        if verbose:
            print(display)
        return print_out

    def evaluate(self, individual: Sequence[bool]):
        """Evaluate ``individual`` and return the score and predictions."""

        return fitness(
            train=self.train,
            test=self.test,
            columns=self.cols,
            ind=individual,
            target=self.target,
            pipeline=self.pipeline,
            scoring=self.scoring,
            ratio=self.ratio,
            cv=self.cv,
        )

    def score(self, individual: Sequence[bool]) -> float:
        """Return only the penalised score associated with ``individual``."""

        return float(self.evaluate(individual)[0])

    def score_population(self, population: Sequence[Sequence[bool]]) -> list[float]:
        """Vectorised convenience wrapper around :meth:`score`."""

        return [self.score(individual) for individual in population]

    def save(
        self,
        name: str,
        bestInd: Sequence[bool],
        bestTime: timedelta,
        g: int,
        t: timedelta,
        last: int,
        specifics: str,
        out: str,
    ) -> None:
        """Persist the optimisation summary and trained pipeline to disk."""

        results_path = Path(self.path) / "results.txt"
        with results_path.open("w", encoding="utf-8") as f:
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
                cv=self.cv,
            )
            if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
                tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
                string_tmp = f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}" + os.linesep
            elif isinstance(self.pipeline.steps[-1][1], RegressorMixin):
                string_tmp = f"Regression residuals (first 5): {(y_true - y_pred).head().tolist()}" + os.linesep
            else:
                string_tmp = ""
            string = (
                f"Heuristic: {name}" + os.linesep
                + f"Population: {self.N}" + os.linesep
                + f"Generations: {self.Gmax}" + os.linesep
                + f"Generations Performed: {g}" + os.linesep
                + f"Latest Improvement: {last}" + os.linesep
                + f"Latest Improvement (Ratio): {1 - (last / g)}" + os.linesep
                + f"Latest Improvement (Time): {round(bestTime.total_seconds())} ({bestTime})" + os.linesep
                + f"Cross-validation strategy: {str(self.cv)}" + os.linesep
                + specifics
                + f"Method: {method}" + os.linesep
                + f"Best Score: {score_train}" + os.linesep
                + string_tmp
                + f"Best Subset: {bestSubset}" + os.linesep
                + f"Number of Features: {len(bestSubset)}" + os.linesep
                + f"Number of Features (Ratio): {len(bestSubset) / len(self.cols)}" + os.linesep
                + f"Execution Time: {round(t.total_seconds())} ({t})" + os.linesep
                + f"Memory: {psutil.virtual_memory()}"
            )
            f.write(string)

        log_path = Path(self.path) / "log.txt"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(out)

        pipeline_path = Path(self.path) / "pipeline.joblib"
        joblib.dump(self.pipeline, pipeline_path)
