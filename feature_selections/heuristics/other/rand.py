"""Random feature selection baseline."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, createDirectory


class Random(Heuristic):
    """Randomly sample feature subsets for benchmarking."""

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
    ) -> None:
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose, output)
        self.path = Path(self.path) / ("rand" + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def create_population(inds: int, size: int) -> np.ndarray:
        pop = np.zeros((inds, size), dtype=int)
        for i in range(inds):
            num_true = np.random.randint(1, size)
            true_indices = np.random.choice(size, size=num_true, replace=False)
            pop[i, true_indices] = 1
        return pop

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401
        self.save("Random Generation", bestInd, bestTime, g, t, last, "", out)

    def start(self, pid: int):
        """Repeatedly sample and evaluate random subsets."""

        code = "RAND"
        debut = time.time()
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)

        population = self.create_population(inds=self.N, size=self.D)
        scores = self.score_population(population)
        bestScore, bestSubset, bestInd = add(scores=scores, inds=population, cols=self.cols)
        scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, timedelta(seconds=0)

        G = 0
        same = 0
        mean_scores = float(np.mean(scores))
        print_out = self.sprint_(
            print_out=print_out,
            name=code,
            pid=pid,
            maxi=scoreMax,
            best=bestScore,
            mean=mean_scores,
            feats=len(subsetMax),
            time_exe=timedelta(seconds=0),
            time_total=timedelta(seconds=0),
            g=G,
            cpt=0,
            verbose=self.verbose,
        ) + "\n"

        while G < self.Gmax:
            instant = time.time()
            neighbourhood = self.create_population(inds=self.N, size=self.D)
            scores = self.score_population(neighbourhood)
            bestScore, bestSubset, bestInd = add(scores=scores, inds=neighbourhood, cols=self.cols)
            G += 1
            same += 1
            mean_scores = float(np.mean(scores))
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

