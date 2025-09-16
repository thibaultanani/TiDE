import os
import time
import numpy as np
from datetime import timedelta

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import createDirectory, fitness


class SimulatedAnnealing(Heuristic):
    """
    Class that implements Simulated Annealing

    Args:
        p0 (float): probability of accepting a deterioration at the very beginning of the run (high temperature)
        pf (float): probability of accepting a deterioration at the end of the run (low temperature)
    """

    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None,
                 Tmax=None, ratio=None, N=None, Gmax=None, suffix=None, cv=None, verbose=None, output=None,
                 p0=0.8, pf=0.01, seed=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.path = os.path.join(self.path, 'simulated_annealing' + self.suffix)
        createDirectory(self.path)
        self.p0 = float(p0)
        self.pf = float(pf)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _time_exceeded(start_time, Tmax):
        return (Tmax is not None) and ((time.time() - start_time) >= Tmax)

    def _ensure_non_empty(self, ind: np.ndarray) -> np.ndarray:
        if ind.sum() == 0:
            j = int(self.rng.integers(self.D))
            ind[j] = 1
        return ind

    def _random_mask(self) -> np.ndarray:
        ind = (self.rng.random(self.D) < 0.5).astype(int)
        return self._ensure_non_empty(ind)

    def _neighbor(self, ind: np.ndarray, s: float) -> np.ndarray:
        D = self.D
        neigh = ind.copy()
        kmax = max(1, int(np.ceil(np.sqrt(D))))
        k_float = 1.0 + (kmax - 1.0) * (1.0 - float(s))
        k_low = int(np.floor(k_float))
        k_high = min(kmax, k_low + 1)
        frac = k_float - k_low
        K = k_high if (self.rng.random() < frac) else k_low
        if K < 1:
            K = 1
        idxs = self.rng.choice(D, size=K, replace=False)
        neigh[idxs] ^= 1
        return self._ensure_non_empty(neigh)

    def _eval(self, ind: np.ndarray) -> float:
        score = fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                        pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]
        return float(score)

    def _calibrate_temperatures(self, ind0: np.ndarray, score0: float, max_samples: int):
        losses = []
        for _ in range(max_samples):
            cand = self._neighbor(ind0, s=0.0)
            s = self._eval(cand)
            delta = s - score0
            if delta < 0:
                losses.append(-delta)
        d = float(np.median(losses)) if len(losses) else 1e-3
        d = max(d, 1e-6)
        T0 = d / max(1e-6, -np.log(self.p0))
        Tf = d / max(1e-6, -np.log(self.pf))
        if Tf < 1e-9:
            Tf = 1e-9
        if T0 < Tf:
            T0, Tf = Tf, T0
        return T0, Tf

    def specifics(self, bestInd, g, t, last, out):
        string = "p0: " + str(self.p0) + os.linesep + "pf: " + str(self.pf) + os.linesep
        self.save("Simulated Annealing", bestInd, g, t, last, string, out)

    def start(self, pid):
        debut = time.time()
        code = "SA  "
        self.path = os.path.join(self.path)
        createDirectory(self.path)
        print_out = ""
        self.rng = np.random.default_rng()
        ind_cur = self._random_mask()
        score_cur = self._eval(ind_cur)
        ind_best = ind_cur.copy()
        score_best = score_cur
        calib_samples = int(min(5 * (self.N or 20), 200))
        T0, Tf = self._calibrate_temperatures(ind_cur, score_cur, calib_samples)
        G, since_improv = 0, 0
        saved = False
        evals_per_iter = int(self.N) if (self.N is not None and self.N > 0) else max(16, min(64, self.D))
        Gmax = int(self.Gmax) if (self.Gmax is not None and self.Gmax > 0) else 10_000
        while G < Gmax:
            instant = time.time()
            if self.Tmax is not None and self.Tmax > 0:
                s = min(1.0, (time.time() - debut) / self.Tmax)
            else:
                s = min(1.0, G / max(1, Gmax - 1))
            T = T0 * ((Tf / T0) ** s)
            improved_this_iter = False
            scores = []
            for _ in range(evals_per_iter):
                if self._time_exceeded(debut, self.Tmax):
                    break
                cand = self._neighbor(ind_cur, s=s)
                score_cand = self._eval(cand)
                scores.append(score_cand)
                delta = score_cand - score_cur
                accept = (delta >= 0)
                if not accept:
                    prob = np.exp(delta / max(T, 1e-12))
                    if self.rng.random() < prob:
                        accept = True
                if accept:
                    ind_cur = cand
                    score_cur = score_cand
                    if score_cur > score_best:
                        ind_best = ind_cur.copy()
                        score_best = score_cur
                        improved_this_iter = True
                if self._time_exceeded(debut, self.Tmax):
                    break
            G += 1
            since_improv = 0 if improved_this_iter else (since_improv + 1)
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))
            feats = int(ind_best.sum())
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=score_best,
                                     best=score_cur, mean=mean_scores, feats=feats,
                                     time_exe=time_instant, time_total=time_total,
                                     g=G, cpt=since_improv, verbose=self.verbose) + "\n"
            stop = self._time_exceeded(debut, self.Tmax)
            if (G % 10 == 0) or stop or (G == Gmax) or (since_improv == self.D):
                self.specifics(bestInd=ind_best, g=G, t=time_total, last=G - since_improv, out=print_out)
                print_out = ""
                saved = True
                if stop:
                    break
        if not saved:
            self.specifics(bestInd=ind_best, g=G, t=timedelta(seconds=(time.time() - debut)), last=G - since_improv,
                           out=print_out)
        selected_features = [self.cols[i] for i in range(self.D) if ind_best[i] == 1]
        return score_best, ind_best, selected_features, self.pipeline, pid, code, G - since_improv, G
