from __future__ import annotations

import os
import time
import warnings
from tempfile import TemporaryDirectory
from datetime import timedelta
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import beta
from sklearn.base import ClassifierMixin
from sklearn.model_selection import check_cv

from feature_selections.filters import Filter
from feature_selections.heuristics.heuristic import Heuristic, PopulationState
from feature_selections.heuristics.population_based.differential import Differential
from feature_selections.heuristics.single_solution import ForwardSelection
from helper.helper import add, create_directory, create_population, fitness, get_entropy


class Tide(Heuristic):
    """Tournament in Differential Evolution feature selection heuristic.

    Parameters specific to this strategy
    -----------------------------------
    gamma: float | None
        Controls the tournament pressure when choosing the guiding individual.
    CR: float | None
        Optional fixed crossover rate. When provided, uses the classic
        binomial crossover rate instead of the adaptive Beta distribution.
    strat: str | None
        Optional differential evolution mutation strategy label. When
        provided, TiDE uses the corresponding DE mutation operator.
    F: float | None
        Differential weight used by DE-style mutations (only when ``strat`` is set).
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
        stability_ratio=None,
        N=None,
        Gmax=None,
        gamma=None,
        CR=None,
        strat=None,
        F=None,
        filter_init=None,
        filter_init_mode=None,
        sfs_init=None,
        sfs_init_mode=None,
        sffs_init=None,
        entropy=None,
        suffix=None,
        cv=None,
        verbose=None,
        output=None,
        warm_start=None,
        seed=None,
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
            stability_ratio,
            suffix,
            verbose,
            output,
            warm_start=warm_start,
            seed=seed,
        )
        self.gamma = gamma if gamma is not None else 0.8
        if CR is not None:
            if not 0.0 <= float(CR) <= 1.0:
                raise ValueError("CR must be between 0 and 1 when provided.")
            self.CR = float(CR)
        else:
            self.CR = None
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
        if strat is not None:
            strat = str(strat).strip()
            if strat not in valid:
                raise ValueError(f"strat '{strat}' invalid. Choose from {sorted(valid)}.")
        self.strat = strat
        self.F = 1.0 if F is None else float(F)
        self.filter_init_mode = self._resolve_filter_init_mode(filter_init_mode)
        self.sfs_init_mode = self._resolve_sfs_init_mode(sfs_init_mode)
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
            self.sfs_init = True
            self.sfs_str = " + SFS" if self.sfs_init_mode == "sfs" else " + SFFS"
        self.entropy = entropy if entropy is not None else 0.05
        self.path = self.path / ("tide" + self.suffix)
        create_directory(path=self.path)

    @staticmethod
    def _resolve_filter_init_mode(filter_init_mode: str | None) -> str:
        """Return the initialisation protocol used for the filter seed."""

        if filter_init_mode is None:
            return "fast_ranking"
        mode = str(filter_init_mode).strip().lower()
        valid = {"fast_ranking", "cv_majority"}
        if mode not in valid:
            raise ValueError(f"filter_init_mode '{filter_init_mode}' invalid. Choose from {sorted(valid)}.")
        return mode

    @staticmethod
    def _filter_init_mode_label(filter_init_mode: str) -> str:
        """Return a human-readable label for the filter initialisation mode."""

        labels = {
            "fast_ranking": "fast global ranking",
            "cv_majority": "per-fold ranking + majority vote",
        }
        return labels[filter_init_mode]

    @staticmethod
    def _resolve_sfs_init_mode(sfs_init_mode: str | None) -> str:
        """Return the sequential initialisation protocol used by TiDE."""

        if sfs_init_mode is None:
            return "sfs"
        mode = str(sfs_init_mode).strip().lower()
        valid = {"sfs", "sffs"}
        if mode not in valid:
            raise ValueError(f"sfs_init_mode '{sfs_init_mode}' invalid. Choose from {sorted(valid)}.")
        return mode

    @staticmethod
    def _sfs_init_mode_label(sfs_init_mode: str) -> str:
        """Return a human-readable label for the sequential init mode."""

        labels = {
            "sfs": "sequential forward selection",
            "sffs": "sequential floating forward selection",
        }
        return labels[sfs_init_mode]

    def _log_initialisation_step(
        self,
        *,
        state: PopulationState,
        pid: int,
        start_time: float,
        step_start: float,
        score: float,
        candidate: Sequence[bool],
    ) -> None:
        """Record an initialisation candidate as a regular TiDE iteration."""

        candidate_array = np.asarray(candidate, dtype=bool)
        subset = [self.cols[i] for i in range(self.D) if candidate_array[i]]
        time_total = self.elapsed_since(start_time)
        state.update_current(score, subset, candidate_array)
        state.advance()
        state.tracker.observe(score, subset, candidate_array, time_total)
        self.log_generation(
            state=state,
            code="TIDE",
            pid=pid,
            maxi=state.tracker.score,
            best=score,
            mean=score,
            feats=len(subset),
            time_exe=timedelta(seconds=(time.time() - step_start)),
            time_total=time_total,
            entropy=1.0,
        )

    def _score_internal_candidate(self, candidate: Sequence[bool]) -> float:
        """Evaluate ``candidate`` with TiDE's internal selection protocol."""

        return float(
            fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=self.sanitize_individual(candidate),
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio,
                stability_ratio=self.stability_ratio,
                cv=self.cv,
                rng=self._rng,
            )[0]
        )

    def _filter_initialisation_fast_ranking(
        self,
        *,
        state: PopulationState | None = None,
        pid: int = 0,
        start_time: float | None = None,
    ) -> Sequence[int]:
        """Initialise the population with a filter-based ranking of features."""

        local_start_time = time.time() if start_time is None else start_time
        if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
            sorted_features, _ = Filter.anova_selection(df=self.train, target=self.target)
        else:
            sorted_features, _ = Filter.correlation_selection(df=self.train, target=self.target)
        best_score, best_vector = float("-inf"), [0] * self.D
        num_features = [int(round(len(sorted_features) * (val / 100.0))) for val in range(1, 101)]
        num_features = [val for val in num_features if val >= 1]
        num_features = list(dict.fromkeys(num_features))

        step_idx = 0
        while step_idx < len(num_features) and (state is None or state.generation < self.Gmax):
            if self.should_stop(local_start_time):
                break
            step_start = time.time()
            top_k_features = sorted_features[:num_features[step_idx]]
            step_idx += 1
            candidate = [0] * self.D
            for var in top_k_features:
                candidate[self.cols.get_loc(var)] = 1
            score = fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=self.sanitize_individual(candidate),
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=self.ratio,
                stability_ratio=self.stability_ratio,
                cv=self.cv,
                rng=self._rng,
            )[0]
            if state is not None:
                self._log_initialisation_step(
                    state=state,
                    pid=pid,
                    start_time=local_start_time,
                    step_start=step_start,
                    score=float(score),
                    candidate=candidate,
                )
            if score > best_score:
                best_score, best_vector = score, candidate
            if self.should_stop(local_start_time):
                break
        return best_vector

    def _filter_initialisation_cv_majority(
        self,
        *,
        state: PopulationState | None = None,
        pid: int = 0,
        start_time: float | None = None,
    ) -> Sequence[int]:
        """Initialise TiDE with the filter protocol based on CV majority vote."""

        local_start_time = time.time() if start_time is None else start_time
        X = self.train.drop(columns=[self.target])
        y = self.train[self.target]
        splitter = check_cv(cv=self.cv, y=y, classifier=isinstance(self.pipeline.steps[-1][1], ClassifierMixin))
        splits = list(splitter.split(X, y))
        majority_threshold = max(1, (len(splits) // 2) + 1)
        fold_best_subsets: list[list[str]] = [[] for _ in splits]
        fold_best_scores = [float("-inf")] * len(splits)
        global_best_score = float("-inf")
        global_best_vector = [0] * self.D

        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            if self.should_stop(local_start_time) or (state is not None and state.generation >= self.Gmax):
                break
            fold_train = self.train.iloc[train_idx].reset_index(drop=True)
            fold_valid = self.train.iloc[valid_idx].reset_index(drop=True)
            if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
                sorted_features, _ = Filter.anova_selection(df=fold_train, target=self.target)
            else:
                sorted_features, _ = Filter.correlation_selection(df=fold_train, target=self.target)

            for feature_count in range(1, len(sorted_features) + 1):
                if self.should_stop(local_start_time) or (state is not None and state.generation >= self.Gmax):
                    break
                step_start = time.time()
                candidate_features = sorted_features[:feature_count]
                candidate = np.zeros(self.D, dtype=int)
                for feature in candidate_features:
                    candidate[self.cols.get_loc(feature)] = 1
                score = float(
                    fitness(
                        train=fold_train,
                        test=fold_valid,
                        columns=self.cols,
                        ind=self.sanitize_individual(candidate),
                        target=self.target,
                        pipeline=self.pipeline,
                        scoring=self.scoring,
                        ratio=self.ratio,
                        stability_ratio=self.stability_ratio,
                        cv=None,
                        rng=self._rng,
                    )[0]
                )
                if score > fold_best_scores[fold_idx]:
                    fold_best_scores[fold_idx] = score
                    fold_best_subsets[fold_idx] = list(candidate_features)
                    feature_votes: dict[str, int] = {}
                    for subset in fold_best_subsets:
                        for feature in subset:
                            feature_votes[feature] = feature_votes.get(feature, 0) + 1
                    consensus_subset = [
                        feature for feature in self.cols if feature_votes.get(feature, 0) >= majority_threshold
                    ]
                    if not consensus_subset and feature_votes:
                        max_votes = max(feature_votes.values())
                        consensus_subset = [
                            feature for feature in self.cols if feature_votes.get(feature, 0) == max_votes
                        ]
                    if not consensus_subset:
                        consensus_subset = list(candidate_features)
                    consensus_vector = np.zeros(self.D, dtype=int)
                    for feature in consensus_subset:
                        consensus_vector[self.cols.get_loc(feature)] = 1
                    consensus_score = self._score_internal_candidate(consensus_vector)
                    if state is not None:
                        self._log_initialisation_step(
                            state=state,
                            pid=pid,
                            start_time=local_start_time,
                            step_start=step_start,
                            score=float(consensus_score),
                            candidate=consensus_vector,
                        )
                    if consensus_score > global_best_score:
                        global_best_score = float(consensus_score)
                        global_best_vector = consensus_vector.tolist()

        return global_best_vector

    def filter_initialisation(
        self,
        *,
        state: PopulationState | None = None,
        pid: int = 0,
        start_time: float | None = None,
    ) -> Sequence[int]:
        """Initialise the population with a filter-based protocol."""

        if self.filter_init_mode == "cv_majority":
            return self._filter_initialisation_cv_majority(state=state, pid=pid, start_time=start_time)
        return self._filter_initialisation_fast_ranking(state=state, pid=pid, start_time=start_time)

    def forward_initialisation(
        self,
        *,
        state: PopulationState | None = None,
        pid: int = 0,
        start_time: float | None = None,
    ) -> Sequence[int]:
        """Initialise the population with the forward selection heuristic."""

        local_start_time = time.time() if start_time is None else start_time
        scoreMax, indMax = -np.inf, np.zeros(self.D, dtype=int)
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
                strat=self.sfs_init_mode,
                seed=self._seed,
            )

            while (state is None or state.generation < self.Gmax) and improvement:
                if self.should_stop(local_start_time):
                    break
                step_start = time.time()
                improvement, _, scoreMax, indMax, timeout = selector._forward_step(
                    start_time=local_start_time,
                    scoreMax=scoreMax,
                    indMax=indMax,
                )
                if timeout:
                    break
                if state is not None:
                    self._log_initialisation_step(
                        state=state,
                        pid=pid,
                        start_time=local_start_time,
                        step_start=step_start,
                        score=float(scoreMax),
                        candidate=indMax,
                    )
                if selector._time_exceeded(local_start_time, self.Tmax):
                    break
        return indMax.tolist()

    def mutate(self, P: Sequence[Sequence[bool]], n_ind: int, current: int, tbest: int) -> list[int]:
        """Mutate an individual following the TiDE mutation operator."""

        selected = self._rng.choice([i for i in range(len(P)) if i != current and i != tbest], 2, replace=False)
        Xr1, Xr2, Xr3 = P[tbest], P[selected[0]], P[selected[1]]
        mutant = []
        for chromosome in range(n_ind):
            if Xr2[chromosome] == Xr3[chromosome]:
                mutant.append(Xr1[chromosome])
            else:
                mutant.append(Xr2[chromosome])
        return mutant

    def _strategy_map(self):
        return {
            "rand/1": lambda pop, n, F, i, best: Differential._mutate_rand(self, pop, n, F, i),
            "best/1": lambda pop, n, F, i, best: Differential._mutate_best(self, pop, n, F, i, best),
            "current-to-rand/1": lambda pop, n, F, i, best: Differential._mutate_current_to_rand_1(self, pop, n, F, i),
            "current-to-best/1": lambda pop, n, F, i, best: Differential._mutate_current_to_best_1(self, pop, n, F, i, best),
            "rand-to-best/1": lambda pop, n, F, i, best: Differential._mutate_rand_to_best_1(self, pop, n, F, i, best),
            "rand/2": lambda pop, n, F, i, best: Differential._mutate_rand_2(self, pop, n, F, i),
            "best/2": lambda pop, n, F, i, best: Differential._mutate_best_2(self, pop, n, F, i, best),
            "current-to-rand/2": lambda pop, n, F, i, best: Differential._mutate_current_to_rand_2(self, pop, n, F, i),
            "current-to-best/2": lambda pop, n, F, i, best: Differential._mutate_current_to_best_2(self, pop, n, F, i, best),
            "rand-to-best/2": lambda pop, n, F, i, best: Differential._mutate_rand_to_best_2(self, pop, n, F, i, best),
        }

    def crossover(self, n_ind: int, ind: Sequence[int], mutant: Sequence[int], cross_proba: float) -> np.ndarray:
        """Perform a binomial crossover between parent and mutant."""

        cross_points = self._rng.random(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = int(self._rng.integers(0, n_ind))
        child[jrand] = mutant[jrand]
        return child

    def tournament(self, scores: Sequence[float], entropy: float, gamma: float) -> int:
        """Return the index of the best individual selected by tournament."""

        p = (1 - entropy) * (1 - gamma) + gamma
        nb_scores = max(2, int(len(scores) * p))
        nb_scores = min(len(scores), nb_scores)
        idx = self._rng.choice(len(scores), size=nb_scores, replace=False)
        selected_scores = [scores[i] for i in idx]
        score_max = float(np.amax(selected_scores))
        tied_indices = [idx_i for idx_i, score in zip(idx, selected_scores) if score == score_max]
        return int(self._rng.choice(tied_indices))

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
        string = (
            "Gamma: "
            + str(self.gamma)
            + os.linesep
            + "Filter init mode: "
            + self.filter_init_mode
            + " ("
            + self._filter_init_mode_label(self.filter_init_mode)
            + ")"
            + os.linesep
            + "Sequential init mode: "
            + self.sfs_init_mode
            + " ("
            + self._sfs_init_mode_label(self.sfs_init_mode)
            + ")"
            + os.linesep
            + "Crossover rate: "
            + (str(self.CR) if self.CR is not None else "adaptive")
            + os.linesep
            + "Mutation strategy: "
            + (self.strat if self.strat is not None else "tide")
            + os.linesep
            + ("F factor: " + str(self.F) + os.linesep if self.strat is not None else "")
        )
        self.save(name, bestInd, bestTime, g, t, last, string, out)

    def _score_individual(self, ind: Sequence[bool]) -> float:
        """Evaluate an individual and return its penalised score."""

        return fitness(
            train=self.train,
            test=self.test,
            columns=self.cols,
            ind=self.sanitize_individual(ind),
            target=self.target,
            pipeline=self.pipeline,
            scoring=self.scoring,
            ratio=self.ratio,
            stability_ratio=self.stability_ratio,
            cv=self.cv,
            rng=self._rng,
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
        self.reset_rng()
        self.reset_tracking()
        self.seed_full_subset_tracking()

        state = PopulationState.from_best(float("-inf"), [], np.zeros(self.D, dtype=bool))

        population = create_population(inds=self.N, size=self.D, rng=self._rng).astype(bool)
        next_slot = 0
        r_filter, r_forward = None, None
        warm_vector = self._warm_start_mask.copy() if self._warm_start_mask is not None else None
        if self.filter_init and next_slot < self.N:
            r_filter = self.filter_initialisation(state=state, pid=pid, start_time=start_time)
            population[next_slot] = np.asarray(r_filter, dtype=bool)
            next_slot += 1
        if self.sfs_init and next_slot < self.N:
            r_forward = self.forward_initialisation(state=state, pid=pid, start_time=start_time)
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
        if state.tracker.score == float("-inf"):
            state = PopulationState.from_best(bestScore, bestSubset, bestInd)
        else:
            time_total = self.elapsed_since(start_time)
            state.update_current(bestScore, bestSubset, bestInd)
            state.tracker.observe(bestScore, bestSubset, bestInd, time_total)

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

        strategy = self._strategy_map()[self.strat] if self.strat is not None else None
        while state.generation < self.Gmax:
            instant = time.time()
            for i in range(self.N):
                if strategy is None:
                    tbest = self.tournament(scores=scores, entropy=entropy, gamma=self.gamma)
                    Vi = self.mutate(P=population, n_ind=self.D, current=i, tbest=tbest)
                else:
                    Vi = strategy(population, self.D, self.F, i, int(np.argmax(scores)))
                if self.CR is None:
                    score_i = max(scores[i], 0)
                    alpha, beta_param = (2 - score_i) * 2, (1 + score_i) * 2
                    CR = beta.rvs(alpha, beta_param, random_state=self._rng)
                else:
                    CR = self.CR
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
                population = create_population(inds=self.N, size=self.D, rng=self._rng).astype(bool)
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

        results_path = self.path / "results.txt"
        pending_logs = state.flush()
        if pending_logs or not results_path.exists():
            self.specifics(
                bestInd=state.tracker.individual,
                bestTime=state.tracker.time_found,
                g=state.generation,
                t=self.elapsed_since(start_time),
                last=state.last_improvement,
                out=pending_logs,
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
