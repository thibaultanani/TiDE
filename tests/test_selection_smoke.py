"""Smoke tests ensuring each feature-selection strategy executes end-to-end."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any, Sequence

import numpy as np
import pytest

from feature_selections.filters.filter import Filter
from feature_selections.heuristics.other.rand import Random
from feature_selections.heuristics.population_based.differential import Differential
from feature_selections.heuristics.population_based.genetic import Genetic
from feature_selections.heuristics.population_based.mbde import Nmbde
from feature_selections.heuristics.population_based.pbil import Pbil
from feature_selections.heuristics.population_based.tide import Tide
from feature_selections.heuristics.single_solution.backward import BackwardSelection
from feature_selections.heuristics.single_solution.forward import ForwardSelection
from feature_selections.heuristics.single_solution.local import LocalSearch
from feature_selections.heuristics.single_solution.simulated import SimulatedAnnealing
from feature_selections.heuristics.single_solution.tabu import Tabu


def _assert_selection_result(result: Sequence[Any], selector, pid: int) -> None:
    score, mask, subset, best_time, pipeline, pid_returned, code, last_improv, generations = result

    mask_array = np.asarray(mask, dtype=bool)
    assert mask_array.shape == (selector.D,)
    assert isinstance(score, (float, int))
    assert isinstance(best_time, timedelta)
    assert pipeline is selector.pipeline
    assert pid_returned == pid
    assert isinstance(code, str) and code
    assert 0 <= last_improv <= generations
    assert set(subset).issubset(set(selector.cols))


@pytest.mark.parametrize(
    "method",
    [
        "correlation",
        "anova",
        "mutual information",
        "mrmr",
        "surf",
    ],
)
def test_filter_strategies_execute(method, dataset, pipeline_factory, tmp_path):
    selector = Filter(
        name=f"filter_{method}_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        Tmax=1,
        ratio=0.1,
        Gmax=5,
        suffix="",
        verbose=False,
        output=str(tmp_path),
        method=method,
    )

    result = selector.start(pid=42)
    _assert_selection_result(result, selector, pid=42)


_HEURISTICS = [
    (
        ForwardSelection,
        dict(Gmax=3, Tmax=1, ratio=0.1, N=4, strat="sfs"),
    ),
    (
        BackwardSelection,
        dict(Gmax=3, Tmax=1, ratio=0.1, N=4, strat="sbs"),
    ),
    (
        LocalSearch,
        dict(Gmax=3, Tmax=1, ratio=0.1, N=4, size=4, nb=1),
    ),
    (
        SimulatedAnnealing,
        dict(Gmax=3, Tmax=1, ratio=0.1, N=4, p0=0.8, pf=0.1),
    ),
    (
        Tabu,
        dict(Gmax=3, Tmax=1, ratio=0.1, N=4, size=4, nb=1),
    ),
    (
        Random,
        dict(Gmax=3, Tmax=1, ratio=0.1, N=4),
    ),
    (
        Genetic,
        dict(Gmax=2, Tmax=1, ratio=0.1, N=4, mutation=1, entropy=0.01),
    ),
    (
        Differential,
        dict(Gmax=2, Tmax=1, ratio=0.1, N=4, F=0.5, CR=0.3, strat="rand/1", entropy=0.01),
    ),
    (
        Nmbde,
        dict(Gmax=2, Tmax=1, ratio=0.1, N=4, Fmax=0.6, Fmin=0.2, CR=0.2, b=5.0, entropy=0.01, strat=False),
    ),
    (
        Pbil,
        dict(Gmax=2, Tmax=1, ratio=0.1, N=6, LR=0.1, MP=0.0, MS=0.1, n=1, entropy=0.01),
    ),
    (
        Tide,
        dict(Gmax=2, Tmax=1, ratio=0.1, N=6, gamma=0.5, filter_init=False, sfs_init=False, entropy=0.01),
    ),
]


@pytest.mark.parametrize("heuristic_cls, params", _HEURISTICS)
def test_heuristic_strategies_execute(heuristic_cls, params, dataset, pipeline_factory, tmp_path):
    selector = heuristic_cls(
        name=f"{heuristic_cls.__name__.lower()}_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        drops=None,
        scoring=None,
        suffix="",
        verbose=False,
        output=str(tmp_path),
        cv=None,
        **params,
    )

    result = selector.start(pid=7)
    _assert_selection_result(result, selector, pid=7)
