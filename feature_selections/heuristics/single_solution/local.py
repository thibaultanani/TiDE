"""Local search heuristic."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from utility.utility import add, create_directory, create_population, diversification


class LocalSearch(Heuristic):
    """Iterative improvement via neighbourhood exploration.

    Parameters specific to this strategy
    -----------------------------------
    size: int | None
        Number of neighbours sampled at each iteration (defaults to ``N``).
    nb: int | None
        Diversification radius; negative values enable the power-law schedule
        from the original implementation.
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
        warm_start=None,
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
        self.size = size if size is not None else self.N
        self.nb = nb if nb is not None else -1
        self.path = Path(self.path) / ("local" + self.suffix)
        create_directory(path=self.path)

    def specifics(self, bestInd, bestTime, g, t, last, out) -> None:  # noqa: D401
        string = "Disruption Rate: " + str(self.nb) + "\n"
        self.save("Local Search", bestInd, bestTime, g, t, last, string, out)

    def start(self, pid: int):
        """Run the local search using diversification neighbourhoods."""

        code = "LS  "
        debut = time.time()
        create_directory(path=self.path)
        print_out = ""
        self.reset_rng()

        population = create_population(inds=self.N, size=self.D, rng=self._rng).astype(bool)
        warm_mask = self._warm_start_mask.copy() if self._warm_start_mask is not None else None
        if warm_mask is not None:
            population[0] = warm_mask
        scores = self.score_population(population)
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
        if warm_mask is not None:
            warm_score = self.score(warm_mask)
            if warm_score >= bestScore:
                bestScore = warm_score
                bestSubset = self.warm_start_features
                bestInd = warm_mask
        scoreMax, subsetMax, indMax, timeMax = bestScore, bestSubset, bestInd, timedelta(seconds=0)

        G = 0
        same = 0
        while G < self.Gmax:
            instant = time.time()
            neighbourhood = [
                diversification(individual=indMax.tolist(), distance=self.nb, rng=self._rng)
                for _ in range(self.N)
            ]
            neighbourhood = np.asarray(neighbourhood)
            scores = [self.score(ind) for ind in neighbourhood]
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
