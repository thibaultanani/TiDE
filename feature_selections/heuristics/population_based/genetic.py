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
from utility.utility import add, createDirectory, create_population, get_entropy, random_int_power


class Genetic(Heuristic):
    """Classic generational genetic algorithm with rank/roulette selection."""

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
        self.path = Path(self.path) / ("genetic" + self.suffix)
        createDirectory(path=self.path)

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
        debut = time.time()
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)

        population = [np.array(ind, copy=True) for ind in create_population(inds=self.N, size=self.D)]
        scores = self.score_population(population)
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
        scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, timedelta(seconds=0)

        G = 0
        same = 0
        while G < self.Gmax:
            instant = time.time()
            population, scores = self.rank_selection(population, scores, self.N)
            for _ in range(self.N):
                parent1, parent2 = self.roulette_selection(population, scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, self.mutation)
                child_array = np.asarray(child)
                child_score = self.score(child_array)
                scores.append(child_score)
                population.append(child_array)

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
                population = [np.array(ind, copy=True) for ind in create_population(inds=self.N, size=self.D)]
                population[0] = indMax
                scores = self.score_population(population)

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

