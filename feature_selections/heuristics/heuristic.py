from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import psutil
from sklearn.base import ClassifierMixin, RegressorMixin

from feature_selections import FeatureSelection
from utility import add, create_population, fitness


def _as_bool_array(individual: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Return a boolean copy of ``individual`` regardless of its original type."""

    return np.asarray(individual, dtype=bool).copy()


@dataclass
class BestTracker:
    """Keep track of the globally best solution discovered so far."""

    score: float
    subset: list[str]
    individual: np.ndarray
    time_found: timedelta = field(default_factory=timedelta)
    stagnation: int = 0

    def observe(
        self,
        score: float,
        subset: Sequence[str],
        individual: Sequence[bool] | np.ndarray,
        time_total: timedelta,
    ) -> None:
        """Update the tracker when a better ``score`` is observed."""

        if score > self.score:
            self.score = float(score)
            self.subset = list(subset)
            self.individual = _as_bool_array(individual)
            self.time_found = time_total
            self.stagnation = 0
        else:
            self.stagnation += 1


@dataclass
class PopulationState:
    """Container storing bookkeeping information for population heuristics."""

    tracker: BestTracker
    current_score: float
    current_subset: list[str]
    current_individual: np.ndarray
    generation: int = 0
    logs: list[str] = field(default_factory=list)

    @classmethod
    def from_best(
        cls,
        best_score: float,
        best_subset: Sequence[str],
        best_individual: Sequence[bool] | np.ndarray,
    ) -> "PopulationState":
        """Create a state seeded with an initial ``best`` solution."""

        best_array = _as_bool_array(best_individual)
        tracker = BestTracker(
            score=float(best_score),
            subset=list(best_subset),
            individual=best_array.copy(),
        )
        return cls(
            tracker=tracker,
            current_score=float(best_score),
            current_subset=list(best_subset),
            current_individual=best_array,
        )

    def advance(self) -> None:
        """Advance the generation counter."""

        self.generation += 1

    def update_current(
        self,
        score: float,
        subset: Sequence[str],
        individual: Sequence[bool] | np.ndarray,
    ) -> None:
        """Record the best solution of the current generation."""

        self.current_score = float(score)
        self.current_subset = list(subset)
        self.current_individual = _as_bool_array(individual)

    def record(self, line: str) -> None:
        """Append ``line`` to the buffered textual log."""

        self.logs.append(line)

    def flush(self) -> str:
        """Return the buffered log and clear the internal storage."""

        if not self.logs:
            return ""
        output = "\n".join(self.logs) + "\n"
        self.logs.clear()
        return output

    @property
    def last_improvement(self) -> int:
        """Return the generation index of the latest improvement."""

        return max(0, self.generation - self.tracker.stagnation)

    def reset_stagnation(self) -> None:
        """Reset the stagnation counter without altering the tracked best."""

        self.tracker.stagnation = 0


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
        warm_start: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
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
        self.N = N if N is not None else 100

    @abc.abstractmethod
    def start(self, pid: int) -> tuple[Any, ...]:
        """Execute the heuristic optimisation process.

        Implementations must return a 9-item tuple matching the layout used by
        the higher-level experiment harness::

            (score, mask, subset, time_found, pipeline, pid, code,
             last_improvement, generations)

        ``score`` is the penalised objective value, ``mask`` is the boolean
        indicator of selected features, ``subset`` lists the feature names,
        ``time_found`` captures when the current best was discovered,
        ``pipeline`` is the fitted estimator, ``pid`` and ``code`` identify the
        run, and the last two integers provide bookkeeping information for
        convergence tracking.
        """

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
            rng=self._rng,
        )

    def score(self, individual: Sequence[bool]) -> float:
        """Return only the penalised score associated with ``individual``."""

        return float(self.evaluate(individual)[0])

    def score_population(self, population: Sequence[Sequence[bool]]) -> list[float]:
        """Vectorised convenience wrapper around :meth:`score`."""

        return [self.score(individual) for individual in population]

    def initialise_population(
        self,
        *,
        as_list: bool = False,
    ) -> tuple[Sequence[np.ndarray], list[float], PopulationState]:
        """Initialise a random population and return its evaluated state."""

        matrix = create_population(inds=self.N, size=self.D, rng=self._rng).astype(bool)
        if self._warm_start_mask is not None:
            matrix[0] = self._warm_start_mask
        if as_list:
            population: Sequence[np.ndarray] = [np.array(ind, copy=True) for ind in matrix]
        else:
            population = matrix
        scores = self.score_population(population)
        best_score, best_subset, best_individual = add(
            scores=scores, inds=np.asarray(population), cols=self.cols
        )
        if self._warm_start_mask is not None:
            warm_score = self.score(self._warm_start_mask)
            if warm_score >= best_score:
                best_score = warm_score
                best_subset = self.warm_start_features
                best_individual = self._warm_start_mask.copy()
        state = PopulationState.from_best(best_score, best_subset, best_individual)
        return population, scores, state

    def restart_population(
        self,
        best_individual: Sequence[bool] | np.ndarray,
        *,
        as_list: bool = False,
    ) -> tuple[Sequence[np.ndarray], list[float], float, list[str], np.ndarray]:
        """Restart the population while keeping ``best_individual``."""

        matrix = create_population(inds=self.N, size=self.D, rng=self._rng).astype(bool)
        if as_list:
            population: Sequence[np.ndarray] = [np.array(ind, copy=True) for ind in matrix]
            population[0] = _as_bool_array(best_individual)
        else:
            population = matrix
            population[0] = _as_bool_array(best_individual)
        scores = self.score_population(population)
        best_score, best_subset, best_individual = add(
            scores=scores, inds=np.asarray(population), cols=self.cols
        )
        return population, scores, best_score, list(best_subset), _as_bool_array(best_individual)

    def log_generation(
        self,
        state: PopulationState,
        *,
        code: str,
        pid: int,
        maxi: float,
        best: float,
        mean: float,
        feats: int,
        time_exe: timedelta,
        time_total: timedelta,
        entropy: float | None = None,
    ) -> None:
        """Pretty-print and buffer the progression information for ``state``."""

        if entropy is None:
            line = self.sprint_(
                print_out="",
                name=code,
                pid=pid,
                maxi=maxi,
                best=best,
                mean=mean,
                feats=feats,
                time_exe=time_exe,
                time_total=time_total,
                g=state.generation,
                cpt=state.tracker.stagnation,
                verbose=self.verbose,
            )
        else:
            line = self.pprint_(
                print_out="",
                name=code,
                pid=pid,
                maxi=maxi,
                best=best,
                mean=mean,
                feats=feats,
                time_exe=time_exe,
                time_total=time_total,
                entropy=entropy,
                g=state.generation,
                cpt=state.tracker.stagnation,
                verbose=self.verbose,
            )
        state.record(line)

    @staticmethod
    def elapsed_since(start_time: float) -> timedelta:
        """Return the elapsed time since ``start_time`` as a :class:`timedelta`."""

        return timedelta(seconds=(time.time() - start_time))

    def should_stop(self, start_time: float) -> bool:
        """Return ``True`` when the time budget has been exhausted."""

        return (time.time() - start_time) >= self.Tmax

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
            estimator = self.pipeline
            if hasattr(self.pipeline, "steps") and getattr(self.pipeline, "steps", None):
                try:
                    estimator = self.pipeline.steps[-1][1]
                except (AttributeError, IndexError, TypeError):
                    estimator = self.pipeline

            try:
                method = estimator.__class__.__name__
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
            if isinstance(estimator, ClassifierMixin):
                tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
                string_tmp = f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}" + os.linesep
            elif isinstance(estimator, RegressorMixin):
                string_tmp = f"Regression residuals (first 5): {(y_true - y_pred).head().tolist()}" + os.linesep
            else:
                string_tmp = ""
            string = (
                f"Heuristic: {name}" + os.linesep
                + f"Population: {self.N}" + os.linesep
                + f"Generations: {self.Gmax}" + os.linesep
                + f"Generations Performed: {g}" + os.linesep
                + f"Latest Improvement: {last}" + os.linesep
                + f"Latest Improvement (Ratio): {1 - (last / g if g else 0)}" + os.linesep
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
                + f"Random Seed: {self._seed}" + os.linesep
                + f"Memory: {psutil.virtual_memory()}"
            )
            f.write(string)

        log_path = Path(self.path) / "log.txt"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(out)

        pipeline_path = Path(self.path) / "pipeline.joblib"
        joblib.dump(self.pipeline, pipeline_path)
