"""Novel Modified Binary Differential Evolution heuristic."""

from __future__ import annotations

import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, createDirectory, create_population, get_entropy


class Nmbde(Heuristic):
    """Novel Modified Binary Differential Evolution heuristic."""

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
        Fmax: float = 0.8,
        Fmin: float = 0.005,
        CR: float = 0.2,
        strat: bool = False,
        b: float = 20.0,
        entropy: float | None = None,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.Fmax = Fmax
        self.Fmin = Fmin
        self.CR = CR
        self.strat = strat
        self.b = b
        self.entropy = entropy if entropy is not None else 0.05
        self.path = self.path / ("nmbde" + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def _prob_est(mixture: np.ndarray, F: float, b: float) -> np.ndarray:
        """Return probability estimates from the mixture vector."""

        denom = 1.0 + 2.0 * F
        return 1.0 / (1.0 + np.exp(-2.0 * b * (mixture - 0.5) / denom))

    @staticmethod
    def mutate_rand(population: np.ndarray, n_ind: int, F: float, current: int, b: float) -> list[int]:
        """DE/rand/1 mutation strategy."""

        idx = [i for i in range(len(population)) if i != current]
        r1, r2, r3 = np.random.choice(idx, 3, replace=False)
        mixture = population[r1] + F * (population[r2] - population[r3])
        P_vec = Nmbde._prob_est(mixture, F, b)
        return [1 if random.random() < P_vec[i] else 0 for i in range(n_ind)]

    @staticmethod
    def mutate_best(population: np.ndarray, n_ind: int, F: float, current: int, best: int, b: float) -> list[int]:
        """DE/best/1 mutation strategy."""

        idx = [i for i in range(len(population)) if i not in (current, best)]
        r1, r2 = np.random.choice(idx, 2, replace=False)
        mixture = population[best] + F * (population[r1] - population[r2])
        P_vec = Nmbde._prob_est(mixture, F, b)
        return [1 if random.random() < P_vec[i] else 0 for i in range(n_ind)]

    @staticmethod
    def crossover(n_ind: int, ind: np.ndarray, mutant: list[int], cross_proba: float) -> np.ndarray:
        """Perform binomial crossover between ``ind`` and ``mutant``."""

        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401 - doc inherited
        label = "DE/best/1" if self.strat else "DE/rand/1"
        string = (
            f"Fmax: {self.Fmax}" + os.linesep + f"Fmin: {self.Fmin}" + os.linesep + f"Crossover rate: {self.CR}" + os.linesep + f"Bandwidth b: {self.b}"
        ) + os.linesep
        self.save(f"Novel Modified Binary Differential Evolution ({label})", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Run the NM-BDE heuristic until the stopping criteria is met."""

        code = "MBDE"
        start_time = time.time()
        createDirectory(path=self.path)
        np.random.seed(None)

        population, scores, state = self.initialise_population()
        bestScore, bestSubset, bestInd = (
            state.current_score,
            state.current_subset,
            state.current_individual,
        )

        while state.generation < self.Gmax:
            instant = time.time()
            elapsed = time.time() - start_time
            frac = min(elapsed / self.Tmax, 1.0)
            F = self.Fmax - (self.Fmax - self.Fmin) * frac
            best_idx = int(np.argmax(scores))
            for i in range(self.N):
                if self.strat:
                    mutant = self.mutate_best(population, self.D, F, i, best_idx, self.b)
                else:
                    mutant = self.mutate_rand(population, self.D, F, i, self.b)
                trial = self.crossover(self.D, population[i], mutant, self.CR)
                if not np.array_equal(population[i], trial):
                    score_trial = self.score(trial)
                else:
                    score_trial = scores[i]
                if score_trial >= scores[i]:
                    population[i], scores[i] = trial, score_trial

            bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
            state.update_current(bestScore, bestSubset, bestInd)
            state.advance()
            mean_scores = float(np.mean(scores))
            entropy = get_entropy(pop=population)
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = self.elapsed_since(start_time)

            state.tracker.observe(bestScore, bestSubset, bestInd, time_total)
            self.log_generation(
                state=state,
                code=code,
                pid=pid,
                maxi=state.tracker.score,
                best=bestScore,
                mean=mean_scores,
                feats=len(state.tracker.subset),
                time_exe=time_instant,
                time_total=time_total,
                entropy=entropy,
            )

            if entropy < self.entropy:
                state.reset_stagnation()
                population, scores, bestScore, bestSubset, bestInd = self.restart_population(
                    state.tracker.individual, as_list=False
                )
                state.update_current(bestScore, bestSubset, bestInd)

            stop = self.should_stop(start_time)
            if state.generation % 10 == 0 or state.generation == self.Gmax or stop:
                self.specifics(
                    bestInd=state.tracker.individual,
                    bestTime=state.tracker.time_found,
                    g=state.generation,
                    t=self.elapsed_since(start_time),
                    last=state.last_improvement,
                    out=state.flush(),
                )
                if stop:
                    break

        return (
            state.tracker.score,
            state.tracker.individual,
            state.tracker.subset,
            state.tracker.time_found,
            self.pipeline,
            pid,
            code,
            state.last_improvement,
            state.generation,
        )

