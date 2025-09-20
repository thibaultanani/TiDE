import os
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, fitness


class ForwardSelection(Heuristic):
    """
    Implements Sequential Forward Selection (SFS) and Sequential Floating Forward Selection (SFFS).
    Choose via `strat`: "sfs" or "sffs".
    """

    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, suffix=None, cv=None, verbose=None, output=None, strat=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.selected_features = []  # initially empty subset
        self.path = os.path.join(self.path, 'forward_selection' + self.suffix)
        createDirectory(path=self.path)
        self.strat = (strat or "sffs").strip().lower()
        if self.strat not in {"sfs", "sffs"}:
            raise ValueError(f"Unknown strat '{strat}'. Expected 'sfs' or 'sffs'.")

    def specifics(self, bestInd, bestTime, g, t, last, out):
        label = "Sequential Forward Selection (SFS)" if self.strat == "sfs" else "Sequential Floating Forward Selection (SFFS)"
        self.save(label, bestInd, bestTime, g, t, last, "", out)

    @staticmethod
    def _time_exceeded(start_time, Tmax):
        if Tmax is None:
            return False
        return (time.time() - start_time) >= Tmax

    def _build_indicator(self, features):
        indicator = np.zeros(self.D, dtype=int)
        for var in features:
            indicator[self.cols.get_loc(var)] = 1
        return indicator

    def _evaluate_features(self, features):
        indicator = self._build_indicator(features)
        score = fitness(
            train=self.train,
            test=self.test,
            columns=self.cols,
            ind=indicator,
            target=self.target,
            pipeline=self.pipeline,
            scoring=self.scoring,
            ratio=self.ratio,
            cv=self.cv
        )[0]
        return indicator, score

    def _forward_step(self, scoreMax, indMax, start_time):
        """
        One SFS step (add only). Returns (improvement, selected_features, scoreMax, indMax, timeout).
        Evaluates candidates and can exit early if time limit exceeded.
        """
        improvement = False
        timeout = False
        best_feature_to_add = None
        for feature in self.cols:
            if feature in self.selected_features:
                continue
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
            candidate_features = self.selected_features + [feature]
            indicator, score = self._evaluate_features(candidate_features)
            if score > scoreMax:
                scoreMax, indMax = score, indicator
                best_feature_to_add = feature
                improvement = True
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
        if best_feature_to_add is not None:
            self.selected_features.append(best_feature_to_add)
        return improvement, self.selected_features, scoreMax, indMax, timeout

    def _backward_step(self, scoreMax, indMax, start_time):
        """
        One backward (floating) step that removes the best candidate feature if it improves the score.
        Returns (improvement, selected_features, scoreMax, indMax, timeout).
        """
        improvement = False
        timeout = False
        best_feature_to_remove = None
        for feature in list(self.selected_features):
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
            candidate_features = [f for f in self.selected_features if f != feature]
            indicator, score = self._evaluate_features(candidate_features)
            if score > scoreMax:
                scoreMax, indMax = score, indicator
                best_feature_to_remove = feature
                improvement = True
            if self._time_exceeded(start_time, self.Tmax):
                timeout = True
                break
        if best_feature_to_remove is not None:
            self.selected_features.remove(best_feature_to_remove)
        return improvement, self.selected_features, scoreMax, indMax, timeout

    def _forward_backward_step(self, scoreMax, indMax, start_time):
        """
        One SFFS step (add then conditional remove). Returns (improvement, selected_features, scoreMax, indMax, timeout).
        Time checks are performed inside both forward and backward phases.
        """
        overall_improvement = False
        fwd_impr, _, scoreMax, indMax, timeout = self._forward_step(scoreMax, indMax, start_time)
        overall_improvement = overall_improvement or fwd_impr
        if timeout:
            return overall_improvement, self.selected_features, scoreMax, indMax, True
        bwd_impr, _, scoreMax, indMax, timeout = self._backward_step(scoreMax, indMax, start_time)
        overall_improvement = overall_improvement or bwd_impr
        return overall_improvement, self.selected_features, scoreMax, indMax, timeout

    def start(self, pid):
        code = "SFS " if self.strat == "sfs" else "SFFS"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, indMax, time_debut = -np.inf, np.zeros(self.D, dtype=int), debut
        G, same_since_improv = 0, 0
        improvement = True
        while G < self.Gmax and improvement:
            instant = time.time()
            if self.strat == "sfs":
                improvement, self.selected_features, scoreMax, indMax, timeout = self._forward_step(
                    scoreMax=scoreMax,
                    indMax=indMax,
                    start_time=debut
                )
            else:
                improvement, self.selected_features, scoreMax, indMax, timeout = self._forward_backward_step(
                    scoreMax=scoreMax,
                    indMax=indMax,
                    start_time=debut
                )
            G += 1
            if improvement:
                same_since_improv = 0
            else:
                same_since_improv += 1
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=scoreMax,
                                     mean=scoreMax, feats=len(self.selected_features), time_exe=time_instant,
                                     time_total=time_debut, g=G, cpt=same_since_improv, verbose=self.verbose) + "\n"
            stop = timeout or self._time_exceeded(debut, self.Tmax)
            if G % 10 == 0 or G == self.Gmax or stop or same_since_improv == self.D or (not improvement):
                self.specifics(bestInd=indMax, bestTime=time_debut, g=G, t=timedelta(seconds=(time.time() - debut)),
                               last=G - same_since_improv, out=print_out)
                print_out = ""
                if stop:
                    break
        return (
            scoreMax,
            indMax,
            self.selected_features,
            time_debut,
            self.pipeline,
            time_debut,
            pid,
            code,
            G - same_since_improv,
            G,
        )
