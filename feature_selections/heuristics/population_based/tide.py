from __future__ import annotations

import os
import random
import time
import warnings
from tempfile import TemporaryDirectory
from datetime import timedelta
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import beta
from sklearn.base import ClassifierMixin

from feature_selections.filters import Filter
from feature_selections.heuristics.heuristic import Heuristic, PopulationState
from feature_selections.heuristics.single_solution import ForwardSelection
from utility.utility import add, create_directory, create_population, fitness, get_entropy


class Tide(Heuristic):
    """Tournament in Differential Evolution feature selection heuristic.

    Parameters specific to this strategy
    -----------------------------------
    gamma: float | None
        Controls the tournament pressure when choosing the guiding individual.
    filter_init: bool | None
        When ``True`` seed the initial population with the best filter ranking.
    sfs_init: bool | None
        When ``True`` seed the initial population with the SFS solution.
    entropy: float | None
        Population entropy threshold triggering diversification.
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
        gamma=None,
        filter_init=None,
        sfs_init=None,
        sffs_init=None,
        entropy=None,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
        warm_start=None,
    ):
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
        self.gamma = gamma if gamma is not None else 0.8
        if filter_init is False:
            self.filter_init, self.filter_str = False, ""
        else:
            self.filter_init = True
            if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
                self.filter_str = " + ANOVA"
            else:
                self.filter_str = " + CORRELATION"
        if sfs_init is None and sffs_init is not None:
            warnings.warn(
                "'sffs_init' is deprecated, use 'sfs_init' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            sfs_init = sffs_init
        if sfs_init is False:
            self.sfs_init, self.sfs_str = False, ""
        else:
            self.sfs_init, self.sfs_str = True, " + SFS"
        self.entropy = entropy if entropy is not None else 0.05
        self.path = self.path / ("tide" + self.suffix)
        create_directory(path=self.path)

    def filter_initialisation(self) -> Sequence[int]:
        """Initialise the population with a filter-based ranking of features."""

        debut = time.time()
        if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
            sorted_features, _ = Filter.anova_selection(df=self.train, target=self.target)
        else:
            sorted_features, _ = Filter.correlation_selection(df=self.train, target=self.target)
        best_score, best_vector = -np.inf, [0] * self.D
        num_features = [int(round(len(sorted_features) * (val / 100.0))) for val in range(1, 101)]
        num_features = [val for val in num_features if val >= 1]
        num_features = list(dict.fromkeys(num_features))
        generation = 0
        while generation < self.Gmax and generation < len(num_features):
            top_k_features = sorted_features[:num_features[generation]]
            generation += 1
            candidate = [0] * self.D
            for var in top_k_features:
                candidate[self.cols.get_loc(var)] = 1
            score = fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=candidate,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio,
                cv=self.cv,
            )[0]
            if score > best_score:
                best_score, best_vector = score, candidate
            if time.time() - debut >= self.Tmax:
                break
        return best_vector

    def forward_initialisation(self) -> Sequence[int]:
        """Initialise the population with the forward selection heuristic."""

        debut = time.time()
        scoreMax, indMax = -np.inf, np.zeros(self.D, dtype=int)
        generation = 0
        improvement = True

        with TemporaryDirectory() as tmp_output:
            selector = ForwardSelection(
                name=f"{self.name}_tide_seed",
                target=self.target,
                pipeline=self.pipeline,
                train=self.train,
                test=self.test,
                drops=None,
                scoring=self.scoring,
                Tmax=self.Tmax,
                ratio=self.ratio,
                N=self.N,
                Gmax=self.Gmax,
                suffix=self.suffix,
                cv=self.cv,
                verbose=False,
                output=tmp_output,
                strat="sfs",
            )

            while generation < self.Gmax and improvement:
                improvement, _, scoreMax, indMax, timeout = selector._forward_step(
                    start_time=debut,
                    scoreMax=scoreMax,
                    indMax=indMax,
                )
                if timeout:
                    break

                generation += 1
                if selector._time_exceeded(debut, self.Tmax):
                    break
        return indMax.tolist()

    @staticmethod
    def mutate(P: Sequence[Sequence[bool]], n_ind: int, current: int, tbest: int) -> list[int]:
        """Mutate an individual following the TiDE mutation operator."""

        selected = np.random.choice([i for i in range(len(P)) if i != current and i != tbest], 2, replace=False)
        Xr1, Xr2, Xr3 = P[tbest], P[selected[0]], P[selected[1]]
        mutant = []
        for chromosome in range(n_ind):
            if Xr2[chromosome] == Xr3[chromosome]:
                mutant.append(Xr1[chromosome])
            else:
                mutant.append(Xr2[chromosome])
        return mutant

    @staticmethod
    def crossover(n_ind: int, ind: Sequence[int], mutant: Sequence[int], cross_proba: float) -> np.ndarray:
        """Perform a binomial crossover between parent and mutant."""

        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    @staticmethod
    def tournament(scores: Sequence[float], entropy: float, gamma: float) -> int:
        """Return the index of the best individual selected by tournament."""

        p = (1 - entropy) * (1 - gamma) + gamma
        nb_scores = max(2, int(len(scores) * p))
        selected = random.choices(scores, k=nb_scores)
        score_max = np.amax(selected)
        return scores.index(score_max)

    def specifics(
        self,
        bestInd: Sequence[bool],
        bestTime: timedelta,
        g: int,
        t: timedelta,
        last: int,
        out: str,
    ) -> None:
        """Serialise TiDE-specific metadata alongside the common heuristic summary."""

        name = "Tournament In Differential" + self.filter_str + self.sfs_str
        string = "Gamma: " + str(self.gamma) + os.linesep
        self.save(name, bestInd, bestTime, g, t, last, string, out)

    def _score_individual(self, ind: Sequence[bool]) -> float:
        """Evaluate an individual and return its penalised score."""

        return fitness(
            train=self.train,
            test=self.test,
            columns=self.cols,
            ind=ind,
            target=self.target,
            pipeline=self.pipeline,
            scoring=self.scoring,
            ratio=self.ratio,
            cv=self.cv,
        )[0]

    def _score_population(self, population: Iterable[Sequence[bool]]) -> list[float]:
        """Vectorised scoring utility for a whole population."""

        return [self._score_individual(ind) for ind in population]

    def start(self, pid: int) -> Tuple[float, Sequence[bool], Sequence[str], timedelta, object, int, str, int, int]:
        """Run the TiDE optimisation loop.

        Returns
        -------
        tuple
            A tuple containing the best score, individual, selected subset and
            bookkeeping information consumed by the experiment harness.
        """

        code = "TIDE"
        start_time = time.time()
        create_directory(path=self.path)
        np.random.seed(None)

        population = create_population(inds=self.N, size=self.D).astype(bool)
        next_slot = 0
        r_filter, r_forward = None, None
        warm_vector = self._warm_start_mask.copy() if self._warm_start_mask is not None else None
        if self.filter_init and next_slot < self.N:
            r_filter = self.filter_initialisation()
            population[next_slot] = np.asarray(r_filter, dtype=bool)
            next_slot += 1
        if self.sfs_init and next_slot < self.N:
            r_forward = self.forward_initialisation()
            population[next_slot] = np.asarray(r_forward, dtype=bool)
            next_slot += 1
        if warm_vector is not None and next_slot < self.N:
            population[next_slot] = warm_vector
            next_slot += 1

        scores = self._score_population(population)
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
        if warm_vector is not None:
            warm_score = self.score(warm_vector)
            if warm_score >= bestScore:
                bestScore = warm_score
                bestSubset = self.warm_start_features
                bestInd = warm_vector
        state = PopulationState.from_best(bestScore, bestSubset, bestInd)

        initial_timer = time.time()
        mean_scores = float(np.mean(scores))
        time_instant = timedelta(seconds=(time.time() - initial_timer))
        time_total = self.elapsed_since(start_time)
        entropy = get_entropy(pop=population)
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

        while state.generation < self.Gmax:
            instant = time.time()
            for i in range(self.N):
                tbest = self.tournament(scores=scores, entropy=entropy, gamma=self.gamma)
                Vi = self.mutate(P=population, n_ind=self.D, current=i, tbest=tbest)
                score_i = max(scores[i], 0)
                alpha, beta_param = (2 - score_i) * 2, (1 + score_i) * 2
                CR = beta.rvs(alpha, beta_param)
                Ui = self.crossover(n_ind=self.D, ind=population[i], mutant=Vi, cross_proba=CR)
                if np.array_equal(population[i], Ui):
                    score_trial = scores[i]
                else:
                    score_trial = self._score_individual(Ui)
                if scores[i] <= score_trial:
                    population[i], scores[i] = Ui, score_trial

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
                population = create_population(inds=self.N, size=self.D).astype(bool)
                inserted = 0
                if r_filter is not None and inserted < self.N:
                    population[inserted] = np.asarray(r_filter, dtype=bool)
                    inserted += 1
                if self.sfs_init and r_forward is not None and inserted < self.N:
                    population[inserted] = np.asarray(r_forward, dtype=bool)
                    inserted += 1
                if warm_vector is not None and inserted < self.N:
                    population[inserted] = warm_vector
                    inserted += 1
                if inserted < self.N:
                    population[inserted] = state.tracker.individual.copy()
                scores = self._score_population(population)
                bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(population), cols=self.cols)
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
