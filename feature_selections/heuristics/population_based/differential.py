"""Binary Differential Evolution heuristics."""

from __future__ import annotations

import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, createDirectory, create_population, get_entropy


class Differential(Heuristic):
    """Binary differential evolution with multiple mutation strategies."""

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
        F: float = 1.0,
        CR: float = 0.5,
        strat: str | None = None,
        entropy: float = 0.05,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.F = F
        self.CR = CR
        self.strat = (strat or "rand/1").strip()
        self.entropy = entropy
        valid = {
            "rand/1",
            "best/1",
            "current-to-rand/1",
            "current-to-best/1",
            "rand-to-best/1",
            "rand/2",
            "best/2",
            "current-to-rand/2",
            "current-to-best/2",
            "rand-to-best/2",
        }
        if self.strat not in valid:
            raise ValueError(f"strat '{self.strat}' invalid. Choose from {sorted(valid)}.")
        self.path = Path(self.path) / ("differential" + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def _mutate_rand(pop: np.ndarray, n_ind: int, F: float, current: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i != current]
        r1, r2, r3 = np.random.choice(idx, 3, replace=False)
        mixture = pop[r1] + F * (pop[r2] - pop[r3])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_best(pop: np.ndarray, n_ind: int, F: float, current: int, best: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i not in (current, best)]
        r1, r2 = np.random.choice(idx, 2, replace=False)
        mixture = pop[best] + F * (pop[r1] - pop[r2])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_current_to_rand_1(pop: np.ndarray, n_ind: int, F: float, current: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i != current]
        r1, r2, r3 = np.random.choice(idx, 3, replace=False)
        mixture = pop[current] + F * (pop[r1] - pop[current]) + F * (pop[r2] - pop[r3])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_current_to_best_1(pop: np.ndarray, n_ind: int, F: float, current: int, best: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i not in (current, best)]
        r1, r2 = np.random.choice(idx, 2, replace=False)
        mixture = pop[current] + F * (pop[best] - pop[current]) + F * (pop[r1] - pop[r2])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_rand_to_best_1(pop: np.ndarray, n_ind: int, F: float, current: int, best: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i != current]
        r1, r2, r3 = np.random.choice(idx, 3, replace=False)
        mixture = pop[r1] + F * (pop[best] - pop[current]) + F * (pop[r2] - pop[r3])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_rand_2(pop: np.ndarray, n_ind: int, F: float, current: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i != current]
        r1, r2, r3, r4, r5 = np.random.choice(idx, 5, replace=False)
        mixture = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_best_2(pop: np.ndarray, n_ind: int, F: float, current: int, best: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i not in (current, best)]
        r1, r2, r3, r4 = np.random.choice(idx, 4, replace=False)
        mixture = pop[best] + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[r4])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_current_to_rand_2(pop: np.ndarray, n_ind: int, F: float, current: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i != current]
        r1, r2, r3, r4, r5 = np.random.choice(idx, 5, replace=False)
        mixture = pop[current] + F * (pop[r1] - pop[current]) + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_current_to_best_2(pop: np.ndarray, n_ind: int, F: float, current: int, best: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i not in (current, best)]
        r1, r2, r3, r4 = np.random.choice(idx, 4, replace=False)
        mixture = pop[current] + F * (pop[best] - pop[current]) + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[r4])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def _mutate_rand_to_best_2(pop: np.ndarray, n_ind: int, F: float, current: int, best: int) -> list[int]:
        idx = [i for i in range(len(pop)) if i != current]
        r1, r2, r3, r4, r5 = np.random.choice(idx, 5, replace=False)
        mixture = pop[r1] + F * (pop[best] - pop[current]) + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])
        return [1 if x >= 0.5 else 0 for x in mixture]

    @staticmethod
    def crossover(n_ind: int, ind: np.ndarray, mutant: list[int], cross_proba: float) -> np.ndarray:
        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    def _strategy_map(self) -> Dict[str, Callable[[np.ndarray, int, float, int, int], list[int]]]:
        return {
            "rand/1": lambda pop, n, F, i, best: self._mutate_rand(pop, n, F, i),
            "best/1": lambda pop, n, F, i, best: self._mutate_best(pop, n, F, i, best),
            "current-to-rand/1": lambda pop, n, F, i, best: self._mutate_current_to_rand_1(pop, n, F, i),
            "current-to-best/1": lambda pop, n, F, i, best: self._mutate_current_to_best_1(pop, n, F, i, best),
            "rand-to-best/1": lambda pop, n, F, i, best: self._mutate_rand_to_best_1(pop, n, F, i, best),
            "rand/2": lambda pop, n, F, i, best: self._mutate_rand_2(pop, n, F, i),
            "best/2": lambda pop, n, F, i, best: self._mutate_best_2(pop, n, F, i, best),
            "current-to-rand/2": lambda pop, n, F, i, best: self._mutate_current_to_rand_2(pop, n, F, i),
            "current-to-best/2": lambda pop, n, F, i, best: self._mutate_current_to_best_2(pop, n, F, i, best),
            "rand-to-best/2": lambda pop, n, F, i, best: self._mutate_rand_to_best_2(pop, n, F, i, best),
        }

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401
        string = (
            "F factor: "
            + str(self.F)
            + os.linesep
            + "Crossover rate: "
            + str(self.CR)
            + os.linesep
            + "Mutation strategy: (DE/"
            + str(self.strat)
            + ")"
            + os.linesep
        )
        self.save(f"Differential Evolution (DE/{self.strat})", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Run the differential evolution optimisation loop."""

        code = "DIFF"
        debut = time.time()
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)

        population = create_population(inds=self.N, size=self.D)
        scores = self.score_population(population)
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
        scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, timedelta(seconds=0)

        strategy = self._strategy_map()[self.strat]

        G = 0
        same = 0
        while G < self.Gmax:
            instant = time.time()
            best_idx = int(np.argmax(scores))
            for i in range(self.N):
                mutant = strategy(population, self.D, self.F, i, best_idx)
                trial = self.crossover(self.D, population[i], mutant, self.CR)
                if not np.array_equal(population[i], trial):
                    score_trial = self.score(trial)
                else:
                    score_trial = scores[i]
                if score_trial >= scores[i]:
                    population[i], scores[i] = trial, score_trial
                    bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)

            G += 1
            same += 1
            mean_scores = float(np.mean(scores))
            entropy = get_entropy(pop=population)
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))

            if bestScore > scoreMax:
                same = 0
                scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, time_total

            print_out = self.pprint_(
                print_out=print_out,
                name=code,
                pid=pid,
                maxi=scoreMax,
                best=bestScore,
                mean=mean_scores,
                feats=len(subsetMax),
                time_exe=time_instant,
                time_total=time_total,
                entropy=entropy,
                g=G,
                cpt=same,
                verbose=self.verbose,
            ) + "\n"

            if entropy < self.entropy:
                same = 0
                population = create_population(inds=self.N, size=self.D)
                population[0] = indMax
                scores = self.score_population(population)
                bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)

            stop = (time.time() - debut) >= self.Tmax
            if G % 10 == 0 or G == self.Gmax or stop:
                self.specifics(
                    bestInd=indMax,
                    bestTime=timeMax,
                    g=G,
                    t=timedelta(seconds=(time.time() - debut)),
                    last=G - same,
                    out=print_out,
                )
                print_out = ""
                if stop:
                    break

        return scoreMax, indMax, subsetMax, timeMax, self.pipeline, pid, code, G - same, G

