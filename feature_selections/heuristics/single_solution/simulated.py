"""Simulated annealing feature selection heuristic."""

from __future__ import annotations

import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import create_directory


class SimulatedAnnealing(Heuristic):
    """Simulated annealing with automatic temperature calibration.

    Parameters specific to this strategy
    -----------------------------------
    p0: float
        Initial acceptance probability used to calibrate the starting
        temperature.
    pf: float
        Final acceptance probability driving the cooling schedule target.
    seed: int | None
        Optional random seed passed to the internal generator.
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
        p0: float = 0.8,
        pf: float = 0.01,
        seed=None,
        warm_start=None,
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
            N,
            Gmax,
            Tmax,
            ratio,
            suffix,
            verbose,
            output,
            warm_start=warm_start,
        )
        self.path = Path(self.path) / ("simulated_annealing" + self.suffix)
        create_directory(self.path)
        self.p0 = float(p0)
        self.pf = float(pf)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _time_exceeded(start_time: float, Tmax) -> bool:
        return Tmax is not None and (time.time() - start_time) >= Tmax

    def _ensure_non_empty(self, ind: np.ndarray) -> np.ndarray:
        if ind.sum() == 0:
            ind[self.rng.integers(self.D)] = 1
        return ind

    def _random_mask(self) -> np.ndarray:
        mask = (self.rng.random(self.D) < 0.5).astype(int)
        return self._ensure_non_empty(mask)

    def _neighbor(self, ind: np.ndarray, schedule_progress: float) -> np.ndarray:
        neigh = ind.copy()
        kmax = max(1, int(np.ceil(np.sqrt(self.D))))
        k_float = 1.0 + (kmax - 1.0) * (1.0 - float(schedule_progress))
        k_low = int(np.floor(k_float))
        k_high = min(kmax, k_low + 1)
        frac = k_float - k_low
        flips = k_high if (self.rng.random() < frac) else k_low
        flips = max(1, flips)
        idxs = self.rng.choice(self.D, size=flips, replace=False)
        neigh[idxs] ^= 1
        return self._ensure_non_empty(neigh)

    def _score(self, ind: np.ndarray) -> float:
        return self.score(ind)

    def _calibrate_temperatures(self, ind0: np.ndarray, score0: float, max_samples: int) -> tuple[float, float]:
        losses = []
        for _ in range(max_samples):
            candidate = self._neighbor(ind0, schedule_progress=0.0)
            delta = self._score(candidate) - score0
            if delta < 0:
                losses.append(-delta)
        d = float(np.median(losses)) if losses else 1e-3
        d = max(d, 1e-6)
        T0 = d / max(1e-6, -math.log(self.p0))
        Tf = d / max(1e-6, -math.log(self.pf))
        T0, Tf = max(T0, Tf), min(T0, Tf)
        Tf = max(Tf, 1e-9)
        return T0, Tf

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401
        string = f"p0: {self.p0}" + "\n" + f"pf: {self.pf}" + "\n"
        self.save("Simulated Annealing", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Run the simulated annealing loop."""

        code = "SA  "
        debut = time.time()
        create_directory(self.path)
        print_out = ""
        self.rng = np.random.default_rng()

        if self._warm_start_mask is not None:
            current = self._ensure_non_empty(self._warm_start_mask.astype(int))
        else:
            current = self._random_mask()
        current_score = self._score(current)
        best = current.copy()
        best_score = current_score
        best_time = timedelta(seconds=0)

        calib_samples = int(min(5 * (self.N or 20), 200))
        T0, Tf = self._calibrate_temperatures(current, current_score, calib_samples)

        evals_per_iter = int(self.N) if (self.N is not None and self.N > 0) else max(16, min(64, self.D))
        Gmax = int(self.Gmax) if (self.Gmax is not None and self.Gmax > 0) else 10_000

        G = 0
        stagnation = 0
        saved = False

        while G < Gmax:
            instant = time.time()
            progress = min(1.0, (time.time() - debut) / self.Tmax) if self.Tmax else min(1.0, G / max(1, Gmax - 1))
            temperature = T0 * ((Tf / T0) ** progress)
            improved = False
            scores = []
            for _ in range(evals_per_iter):
                if self._time_exceeded(debut, self.Tmax):
                    break
                candidate = self._neighbor(current, schedule_progress=progress)
                candidate_score = self._score(candidate)
                scores.append(candidate_score)
                delta = candidate_score - current_score
                accept = delta >= 0 or self.rng.random() < math.exp(delta / max(temperature, 1e-12))
                if accept:
                    current = candidate
                    current_score = candidate_score
                    if current_score > best_score:
                        best = current.copy()
                        best_score = current_score
                        best_time = timedelta(seconds=(time.time() - debut))
                        improved = True
                if self._time_exceeded(debut, self.Tmax):
                    break

            G += 1
            stagnation = 0 if improved else stagnation + 1
            mean_scores = float(np.mean(scores)) if scores else current_score
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))
            feats = int(best.sum())
            print_out = self.sprint_(
                print_out=print_out,
                name=code,
                pid=pid,
                maxi=best_score,
                best=current_score,
                mean=mean_scores,
                feats=feats,
                time_exe=time_instant,
                time_total=time_total,
                g=G,
                cpt=stagnation,
                verbose=self.verbose,
            ) + "\n"

            stop = self._time_exceeded(debut, self.Tmax)
            if (G % 10 == 0) or stop or (G == Gmax) or (stagnation == self.D):
                self.specifics(
                    bestInd=best,
                    bestTime=best_time,
                    g=G,
                    t=time_total,
                    last=G - stagnation,
                    out=print_out,
                )
                print_out = ""
                saved = True
                if stop:
                    break

        if not saved:
            self.specifics(
                bestInd=best,
                bestTime=best_time,
                g=G,
                t=timedelta(seconds=(time.time() - debut)),
                last=G - stagnation,
                out=print_out,
            )

        selected_features = [self.cols[i] for i in range(self.D) if best[i] == 1]
        return best_score, best, selected_features, best_time, self.pipeline, pid, code, G - stagnation, G
