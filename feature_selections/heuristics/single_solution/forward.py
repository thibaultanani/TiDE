"""Sequential forward selection heuristics."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import create_directory, fitness


class ForwardSelection(Heuristic):
    """Sequential (floating) forward selection.

    Parameters specific to this strategy
    -----------------------------------
    strat: str | None
        ``'sfs'`` enables vanilla forward selection, while ``'sffs'`` activates
        the floating variant with backward pruning steps.
    """

    def __init__(
        self,
        name,
        target,
        pipeline,
        train,
        test=None,
        drops=None,
        scoring=None,
        Tmax=None,
        ratio=None,
        N=None,
        Gmax=None,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
        strat=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.selected_features: List[str] = []
        self.strat = (strat or "sffs").strip().lower()
        if self.strat not in {"sfs", "sffs"}:
            raise ValueError(f"Unknown strat '{strat}'. Expected 'sfs' or 'sffs'.")
        self.path = Path(self.path) / ("forward_selection" + self.suffix)
        create_directory(path=self.path)

    @staticmethod
    def _time_exceeded(start_time: float, Tmax: float | None) -> bool:
        return Tmax is not None and (time.time() - start_time) >= Tmax

    def _evaluate_candidate(self, feature_set: Sequence[str]) -> Tuple[float, np.ndarray]:
        candidate = np.zeros(self.D, dtype=int)
        for var in feature_set:
            candidate[self.cols.get_loc(var)] = 1

        score = fitness(
            train=self.train,
            test=self.test,
            columns=self.cols,
            ind=candidate,
            target=self.target,
            pipeline=self.pipeline,
            scoring=self.scoring,
            ratio=self.ratio,
            cv=self.cv,
        )[0]
        return score, candidate

    def _forward_step(
        self,
        start_time: float,
        scoreMax: float,
        indMax: np.ndarray | int,
    ) -> Tuple[bool, List[str], float, np.ndarray | int, bool]:
        improvement = False
        timeout = False
        best_feature = None
        for feature in self.cols:
            if feature in self.selected_features:
                continue
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
            candidate_features = self.selected_features + [feature]
            score, candidate = self._evaluate_candidate(candidate_features)
            if score > scoreMax:
                scoreMax, indMax = score, candidate
                best_feature = feature
                improvement = True
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
        if best_feature is not None:
            self.selected_features.append(best_feature)
        return improvement, self.selected_features, scoreMax, indMax, timeout

    def _backward_step(
        self,
        start_time: float,
        scoreMax: float,
        indMax: np.ndarray | int,
    ) -> Tuple[bool, List[str], float, np.ndarray | int, bool]:
        improvement = False
        timeout = False
        best_to_remove = None
        for feature in list(self.selected_features):
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
            candidate_features = [f for f in self.selected_features if f != feature]
            score, candidate = self._evaluate_candidate(candidate_features)
            if score > scoreMax:
                scoreMax, indMax = score, candidate
                best_to_remove = feature
                improvement = True
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
        if best_to_remove is not None:
            self.selected_features.remove(best_to_remove)
        return improvement, self.selected_features, scoreMax, indMax, timeout

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401
        label = (
            "Sequential Forward Selection (SFS)"
            if self.strat == "sfs"
            else "Sequential Floating Forward Selection (SFFS)"
        )
        self.save(label, bestInd, bestTime, g, t, last, "", out)

    def start(self, pid: int):
        """Execute the forward (floating) selection routine."""

        code = "SFS " if self.strat == "sfs" else "SFFS"
        debut = time.time()
        create_directory(path=self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, indMax = -np.inf, np.zeros(self.D, dtype=int)
        G = 0
        same_since_improv = 0
        improvement = True

        while G < self.Gmax and improvement:
            instant = time.time()
            if self.strat == "sfs":
                improvement, _, scoreMax, indMax, timeout = self._forward_step(debut, scoreMax, indMax)
            else:
                improvement, _, scoreMax, indMax, timeout = self._forward_step(debut, scoreMax, indMax)
                if not timeout and self.selected_features:
                    back_improv, _, scoreMax, indMax, timeout = self._backward_step(debut, scoreMax, indMax)
                    improvement = improvement or back_improv
            G += 1
            same_since_improv = 0 if improvement else same_since_improv + 1
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))
            print_out = self.sprint_(
                print_out=print_out,
                name=code,
                pid=pid,
                maxi=scoreMax,
                best=scoreMax,
                mean=scoreMax,
                feats=len(self.selected_features),
                time_exe=time_instant,
                time_total=time_total,
                g=G,
                cpt=same_since_improv,
                verbose=self.verbose,
            ) + "\n"
            stop = timeout or self._time_exceeded(debut, self.Tmax)
            if G % 10 == 0 or G == self.Gmax or stop or same_since_improv == self.D or (not improvement):
                self.specifics(
                    bestInd=indMax,
                    bestTime=time_total,
                    g=G,
                    t=timedelta(seconds=(time.time() - debut)),
                    last=G - same_since_improv,
                    out=print_out,
                )
                print_out = ""
                if stop:
                    break
        best_time = timedelta(seconds=(time.time() - debut))
        return (
            scoreMax,
            indMax,
            self.selected_features,
            best_time,
            self.pipeline,
            pid,
            code,
            G - same_since_improv,
            G,
        )
