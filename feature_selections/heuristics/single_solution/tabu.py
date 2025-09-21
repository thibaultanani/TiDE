"""Tabu search heuristic."""

from __future__ import annotations

import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, create_directory, create_population, diversification


class Tabu(Heuristic):
    """Tabu search with diversification moves.

    Parameters specific to this strategy
    -----------------------------------
    size: int | None
        Size of the tabu list (defaults to ``N``).
    nb: int | None
        Diversification radius used when generating candidate neighbours;
        negative values activate the power-law move generator.
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
        size=None,
        nb=None,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.size = size if size is not None else self.N
        self.nb = nb if nb is not None else -1
        self.path = Path(self.path) / ("tabu" + self.suffix)
        create_directory(path=self.path)

    @staticmethod
    def _is_in_tabu(individual: np.ndarray, tabu_list: deque[np.ndarray]) -> bool:
        return any(np.array_equal(individual, entry) for entry in tabu_list)

    @staticmethod
    def _insert_tabu(tabu_list: deque[np.ndarray], individual: np.ndarray, size: int) -> deque[np.ndarray]:
        if len(tabu_list) >= size:
            tabu_list.popleft()
        tabu_list.append(individual.copy())
        return tabu_list

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401
        string = "Tabu List Size: " + str(self.size) + "\n" + "Disruption Rate: " + str(self.nb) + "\n"
        self.save("Tabu Search", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Execute the tabu search heuristic."""

        code = "TABU"
        debut = time.time()
        create_directory(path=self.path)
        print_out = ""
        np.random.seed(None)

        population = create_population(inds=self.N, size=self.D)
        scores = self.score_population(population)
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
        scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, timedelta(seconds=0)

        tabu_list: deque[np.ndarray] = deque(maxlen=self.size)
        tabu_list = self._insert_tabu(tabu_list, bestInd, self.size)

        G = 0
        same = 0
        while G < self.Gmax:
            instant = time.time()
            scores: list[float] = []
            neighbourhood = []
            for _ in range(self.N):
                neighbour = np.asarray(diversification(individual=bestInd.tolist(), distance=self.nb))
                if not self._is_in_tabu(neighbour, tabu_list):
                    neighbourhood.append(neighbour)

            if neighbourhood:
                scores = [self.score(ind) for ind in neighbourhood]
                bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(neighbourhood), cols=self.cols)
                tabu_list = self._insert_tabu(tabu_list, bestInd, self.size)

            G += 1
            same += 1
            mean_scores = float(np.mean(scores)) if neighbourhood else float(bestScore)
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))
            if bestScore > scoreMax:
                same = 0
                scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, time_total
            print_out = self.sprint_(
                print_out=print_out,
                name=code,
                pid=pid,
                maxi=scoreMax,
                best=bestScore,
                mean=mean_scores,
                feats=len(subsetMax),
                time_exe=time_instant,
                time_total=time_total,
                g=G,
                cpt=same,
                verbose=self.verbose,
            ) + "\n"

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
