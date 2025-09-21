"""Genetic algorithm for feature selection."""

from __future__ import annotations

import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, create_directory, get_entropy, random_int_power


class Genetic(Heuristic):
    """Classic generational genetic algorithm with rank/roulette selection.

    Parameters specific to this strategy
    -----------------------------------
    mutation: int | None
        Maximum number of bit flips applied during mutation. A negative value
        activates the power-law sampling used in the original implementation.
    entropy: float | None
        Minimum population entropy tolerated before triggering a full restart.
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
        mutation=None,
        entropy=None,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.mutation = mutation if mutation is not None else -1
        self.entropy = entropy if entropy is not None else 0.05
        self.path = self.path / ("genetic" + self.suffix)
        create_directory(path=self.path)

    @staticmethod
    def get_ranks(scores: Sequence[float]) -> list[int]:
        """Return deterministic ranks (1..n) for ``scores`` preserving ties."""

        unique_scores = {score: rank + 1 for rank, score in enumerate(sorted(set(scores)))}
        return [unique_scores[score] for score in scores]

    def rank_selection(
        self,
        population: list[np.ndarray],
        scores: Sequence[float],
        n_select: int,
    ) -> Tuple[list[np.ndarray], list[float]]:
        """Select ``n_select`` individuals proportionally to their rank."""

        ranks = self.get_ranks(scores)
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        selected_indices = np.random.choice(len(population), n_select, p=probabilities, replace=False)
        return [population[i] for i in selected_indices], [scores[i] for i in selected_indices]

    @staticmethod
    def roulette_selection(population: Sequence[np.ndarray], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return two parents sampled proportionally to ``scores``."""

        min_score = min(scores)
        if min_score < 0:
            shifted_scores = [s - min_score + 1e-8 for s in scores]
        else:
            shifted_scores = list(scores)
        total_score = sum(shifted_scores)
        if total_score == 0:
            probabilities = [1 / len(scores)] * len(scores)
        else:
            probabilities = [s / total_score for s in shifted_scores]
        selected_indices = np.random.choice(len(population), 2, p=probabilities, replace=False)
        return population[selected_indices[0]], population[selected_indices[1]]

    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> list[int]:
        """Perform a two-point crossover between ``parent1`` and ``parent2``."""

        p1 = parent1.tolist()
        p2 = parent2.tolist()
        if random.random() < 0.5:
            p1, p2 = p2, p1
        point1 = random.randint(0, len(p1) - 1)
        point2 = random.randint(point1, len(p1) - 1)
        return p1[:point1] + p2[point1:point2] + p1[point2:]

    @staticmethod
    def mutate(individual: Sequence[int], mutation: int) -> list[int]:
        """Flip ``mutation`` random bits (power-law distributed when ``mutation`` < 0)."""

        mutant = list(individual)
        size = len(mutant)
        if mutation >= 0:
            num_moves = random.randint(0, mutation)
        else:
            num_moves = random_int_power(n=size + 1, power=2) - 1
        move_indices = random.sample(range(size), num_moves)
        for idx in move_indices:
            mutant[idx] = 1 - mutant[idx]
        return mutant

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401 - short summary inherited
        string = "Mutation Rate: " + str(self.mutation) + os.linesep
        self.save("Genetic Algorithm", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Run the genetic algorithm loop until the budget is exhausted."""

        code = "GENE"
        start_time = time.time()
        create_directory(path=self.path)
        np.random.seed(None)

        population, scores, state = self.initialise_population(as_list=True)

        while state.generation < self.Gmax:
            instant = time.time()
            population, scores = self.rank_selection(list(population), scores, self.N)
            for _ in range(self.N):
                parent1, parent2 = self.roulette_selection(population, scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, self.mutation)
                child_array = np.asarray(child, dtype=bool)
                child_score = self.score(child_array)
                scores.append(child_score)
                population.append(child_array)

            bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
            state.update_current(bestScore, bestSubset, bestInd)
            state.advance()
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = self.elapsed_since(start_time)
            entropy = get_entropy(pop=population)
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
                    state.tracker.individual, as_list=True
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
