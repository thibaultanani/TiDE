"""Population Based Incremental Learning for feature selection."""

from __future__ import annotations

import os
import random
import time
from copy import copy
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, createDirectory, get_entropy


class Pbil(Heuristic):
    """Population Based Incremental Learning heuristic."""

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
        LR: float = 0.1,
        MP: float = 0.05,
        MS: float = 0.1,
        n: int | None = None,
        entropy: float = 0.05,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.LR = LR
        self.MP = MP
        self.MS = MS
        self.n = n if n is not None else int(self.N * 0.15)
        self.entropy = entropy
        self.path = Path(self.path) / ("pbil" + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def create_probas(size: int) -> list[float]:
        """Create a probability vector initialised at 0.5 for each feature."""

        return [0.5] * size

    @staticmethod
    def top_n_average(population: Sequence[Sequence[int]], scores: Sequence[float], n: int) -> np.ndarray:
        """Return the average of the ``n`` best individuals."""

        top_n_indices = np.argsort(scores)[-n:]
        top_n_vectors = np.asarray([population[i] for i in top_n_indices], dtype=float)
        return np.mean(top_n_vectors, axis=0)

    @staticmethod
    def update_probas(probas: Sequence[float], LR: float, bests: Sequence[float]) -> list[float]:
        """Move ``probas`` towards ``bests`` using the learning rate ``LR``."""

        return [(p * (1.0 - LR)) + (LR * b) for p, b in zip(probas, bests)]

    @staticmethod
    def mutate_probas(probas: Sequence[float], MP: float, MS: float) -> list[float]:
        """Randomly perturb the probability vector to avoid stagnation."""

        mutated = list(probas)
        for i in range(len(mutated)):
            if random.random() < MP:
                mutated[i] = (mutated[i] * (1.0 - MS)) + (MS * random.choice([1.0, 0.0]))
        return mutated

    @staticmethod
    def sample_population(inds: int, size: int, probas: Sequence[float]) -> list[list[int]]:
        """Sample ``inds`` individuals according to ``probas``."""

        population = []
        for _ in range(inds):
            individual = [1 if random.random() < probas[j] else 0 for j in range(size)]
            population.append(individual)
        return population

    def specifics(self, probas, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401 - doc inherited
        string = (
            "Learning Rate: "
            + str(self.LR)
            + os.linesep
            + "Mutation Probabilities: "
            + str(self.MP)
            + os.linesep
            + "Mutation Shift: "
            + str(self.MS)
            + os.linesep
            + "Probabilities Vector: "
            + str([f"{x:.3f}" for x in probas])
            + os.linesep
        )
        self.save("Population Based Incremental Learning", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Run the PBIL loop until the generation or time budget is reached."""

        code = "PBIL"
        debut = time.time()
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)

        probas = self.create_probas(size=self.D)
        saved_proba = copy(probas)
        scoreMax, subsetMax, indMax, timeMax = -np.inf, [], [], timedelta(seconds=0)
        G = 0
        same = 0

        while G < self.Gmax:
            instant = time.time()
            population = self.sample_population(inds=self.N, size=self.D, probas=probas)
            scores = [self.score(ind) for ind in population]
            bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
            G += 1
            same += 1
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))
            entropy = get_entropy(pop=population)

            if bestScore > scoreMax:
                same = 0
                scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, time_total
                saved_proba = copy(probas)

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

            stop = (time.time() - debut) >= self.Tmax
            if G % 10 == 0 or G == self.Gmax or stop:
                self.specifics(
                    probas=saved_proba,
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

            bests = self.top_n_average(population=population, scores=scores, n=self.n)
            probas = self.update_probas(probas=probas, LR=self.LR, bests=bests)
            probas = self.mutate_probas(probas=probas, MP=self.MP, MS=self.MS)

            if entropy < self.entropy:
                same = 0
                probas = self.create_probas(size=self.D)

        return scoreMax, indMax, subsetMax, timeMax, self.pipeline, pid, code, G - same, G

