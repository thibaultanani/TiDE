"""Population Based Incremental Learning for feature selection."""

from __future__ import annotations

import os
import time
from copy import copy
from datetime import timedelta
from typing import Sequence

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic, PopulationState
from utility.utility import add, create_directory, get_entropy


class Pbil(Heuristic):
    """Population Based Incremental Learning heuristic.

    Parameters specific to this strategy
    -----------------------------------
    LR: float
        Learning rate that pulls the probability vector towards the best
        samples.
    MP: float
        Mutation probability applied to each entry of the probability vector.
    MS: float
        Mutation shift magnitude when ``MP`` triggers.
    n: int | None
        Number of elite individuals averaged to update the probabilities
        (defaults to 15% of ``N``).
    entropy: float
        Threshold under which the probability vector is reset to 0.5.
    warm_start_prob: float
        Initial probability assigned to warm-start features (defaults to 0.8).
    warm_start_cold_prob: float
        Initial probability assigned to features outside the warm start when a
        warm start is supplied (defaults to 0.2).
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
        LR: float = 0.1,
        MP: float = 0.05,
        MS: float = 0.1,
        n: int | None = None,
        entropy: float = 0.05,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
        warm_start=None,
        warm_start_prob: float = 0.8,
        warm_start_cold_prob: float = 0.2,
        seed=None,
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
            seed=seed,
        )
        self.LR = LR
        self.MP = MP
        self.MS = MS
        self.n = n if n is not None else int(self.N * 0.15)
        self.entropy = entropy
        self.warm_start_prob = float(np.clip(warm_start_prob, 1e-3, 1 - 1e-3))
        self.warm_start_cold_prob = float(np.clip(warm_start_cold_prob, 1e-3, 1 - 1e-3))
        self.path = self.path / ("pbil" + self.suffix)
        create_directory(path=self.path)

    @staticmethod
    def create_probas(
        size: int,
        warm_mask: np.ndarray | None,
        warm_prob: float,
        cold_prob: float,
    ) -> list[float]:
        """Create a probability vector initialised with optional warm bias."""

        if warm_mask is None:
            return [0.5] * size
        probas = np.full(size, cold_prob, dtype=float)
        probas[warm_mask] = warm_prob
        return probas.tolist()

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

    def mutate_probas(self, probas: Sequence[float], MP: float, MS: float) -> list[float]:
        """Randomly perturb the probability vector to avoid stagnation."""

        mutated = list(probas)
        for i in range(len(mutated)):
            if self._rng.random() < MP:
                mutated[i] = (mutated[i] * (1.0 - MS)) + (MS * self._rng.choice([1.0, 0.0]))
        return mutated

    def sample_population(self, inds: int, size: int, probas: Sequence[float]) -> list[list[int]]:
        """Sample ``inds`` individuals according to ``probas``."""

        population = []
        for _ in range(inds):
            draws = self._rng.random(size)
            individual = [1 if draws[j] < probas[j] else 0 for j in range(size)]
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
        start_time = time.time()
        create_directory(path=self.path)
        self.reset_rng()

        probas = self.create_probas(
            size=self.D,
            warm_mask=self._warm_start_mask,
            warm_prob=self.warm_start_prob,
            cold_prob=self.warm_start_cold_prob,
        )
        saved_proba = copy(probas)
        state = PopulationState.from_best(float("-inf"), [], np.zeros(self.D, dtype=bool))

        while state.generation < self.Gmax:
            instant = time.time()
            population = self.sample_population(inds=self.N, size=self.D, probas=probas)
            scores = [self.score(ind) for ind in population]
            bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
            state.update_current(bestScore, bestSubset, bestInd)
            state.advance()
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = self.elapsed_since(start_time)
            entropy = get_entropy(pop=population)

            previous_best = state.tracker.score
            state.tracker.observe(bestScore, bestSubset, bestInd, time_total)
            if state.tracker.score > previous_best:
                saved_proba = copy(probas)

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

            stop = self.should_stop(start_time)
            if state.generation % 10 == 0 or state.generation == self.Gmax or stop:
                self.specifics(
                    probas=saved_proba,
                    bestInd=state.tracker.individual,
                    bestTime=state.tracker.time_found,
                    g=state.generation,
                    t=self.elapsed_since(start_time),
                    last=state.last_improvement,
                    out=state.flush(),
                )
                if stop:
                    break

            bests = self.top_n_average(population=population, scores=scores, n=self.n)
            probas = self.update_probas(probas=probas, LR=self.LR, bests=bests)
            probas = self.mutate_probas(probas=probas, MP=self.MP, MS=self.MS)

            if entropy < self.entropy:
                state.reset_stagnation()
            probas = self.create_probas(
                size=self.D,
                warm_mask=self._warm_start_mask,
                warm_prob=self.warm_start_prob,
                cold_prob=self.warm_start_cold_prob,
            )

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
