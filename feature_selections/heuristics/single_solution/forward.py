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

    def specifics(self, bestInd, g, t, last, out):
        label = "Sequential Forward Selection (SFS)" if self.strat == "sfs" else "Sequential Floating Forward Selection (SFFS)"
        self.save(label, bestInd, g, t, last, "", out)

    @staticmethod
    def _time_exceeded(start_time, Tmax):
        if Tmax is None:
            return False
        return (time.time() - start_time) >= Tmax

    @staticmethod
    def forward_step(train, test, cols, D, target, pipeline, scoring, ratio, cv,
                     selected_features, scoreMax, indMax, start_time, Tmax):
        """
        One SFS step (add only). Returns (improvement, selected_features, scoreMax, indMax, timeout).
        Evaluates candidates and can exit early if time limit exceeded.
        """
        improvement = False
        timeout = False
        best_feature_to_add = None
        for feature in cols:
            if feature in selected_features:
                continue
            if ForwardSelection._time_exceeded(start_time, Tmax):
                timeout = True
                break
            candidate_features = selected_features + [feature]
            candidate = np.zeros(D, dtype=int)
            for var in candidate_features:
                candidate[cols.get_loc(var)] = 1
            if ForwardSelection._time_exceeded(start_time, Tmax):
                timeout = True
                break
            score = fitness(train=train, test=test, columns=cols, ind=candidate, target=target,
                            pipeline=pipeline, scoring=scoring, ratio=ratio, cv=cv)[0]
            if score > scoreMax:
                scoreMax, indMax = score, candidate
                best_feature_to_add = feature
                improvement = True
            if ForwardSelection._time_exceeded(start_time, Tmax):
                timeout = True
                break
        if best_feature_to_add is not None:
            selected_features.append(best_feature_to_add)
        return improvement, selected_features, scoreMax, indMax, timeout

    @staticmethod
    def forward_backward_step(train, test, cols, D, target, pipeline, scoring, ratio, cv,
                              selected_features, scoreMax, indMax, start_time, Tmax):
        """
        One SFFS step (add then conditional remove). Returns (improvement, selected_features, scoreMax, indMax, timeout).
        Time checks are performed inside both forward and backward phases.
        """
        timeout = False
        overall_improvement = False
        fwd_impr, selected_features, scoreMax, indMax, timeout = ForwardSelection.forward_step(
            train, test, cols, D, target, pipeline, scoring, ratio, cv,
            selected_features, scoreMax, indMax, start_time, Tmax
        )
        overall_improvement = overall_improvement or fwd_impr
        if timeout:
            return overall_improvement, selected_features, scoreMax, indMax, True
        best_feature_to_remove = None
        for feature in list(selected_features):
            if ForwardSelection._time_exceeded(start_time, Tmax):
                timeout = True
                break
            candidate_features = [f for f in selected_features if f != feature]
            candidate = np.zeros(D, dtype=int)
            for var in candidate_features:
                candidate[cols.get_loc(var)] = 1
            if ForwardSelection._time_exceeded(start_time, Tmax):
                timeout = True
                break
            score = fitness(train=train, test=test, columns=cols, ind=candidate, target=target,
                            pipeline=pipeline, scoring=scoring, ratio=ratio, cv=cv)[0]
            if score > scoreMax:
                scoreMax, indMax = score, candidate
                best_feature_to_remove = feature
                overall_improvement = True
            if ForwardSelection._time_exceeded(start_time, Tmax):
                timeout = True
                break
        if best_feature_to_remove is not None:
            selected_features.remove(best_feature_to_remove)
        return overall_improvement, selected_features, scoreMax, indMax, timeout

    def start(self, pid):
        code = "SFS " if self.strat == "sfs" else "SFFS"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, indMax = -np.inf, 0
        G, same_since_improv = 0, 0
        improvement = True
        while G < self.Gmax and improvement:
            instant = time.time()
            if self.strat == "sfs":
                improvement, self.selected_features, scoreMax, indMax, timeout = ForwardSelection.forward_step(
                    train=self.train, test=self.test, cols=self.cols, D=self.D, target=self.target,
                    pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv,
                    selected_features=self.selected_features, scoreMax=scoreMax, indMax=indMax,
                    start_time=debut, Tmax=self.Tmax
                )
            else:
                improvement, self.selected_features, scoreMax, indMax, timeout = ForwardSelection.forward_backward_step(
                    train=self.train, test=self.test, cols=self.cols, D=self.D, target=self.target,
                    pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv,
                    selected_features=self.selected_features, scoreMax=scoreMax, indMax=indMax,
                    start_time=debut, Tmax=self.Tmax
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
                self.specifics(bestInd=indMax, g=G, t=timedelta(seconds=(time.time() - debut)),
                               last=G - same_since_improv, out=print_out)
                print_out = ""
                if stop:
                    break
        return scoreMax, indMax, self.selected_features, self.pipeline, pid, code, G - same_since_improv, G
