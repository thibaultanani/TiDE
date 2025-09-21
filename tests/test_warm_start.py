"""Warm-start behaviour tests for representative feature-selection strategies."""

from __future__ import annotations

import numpy as np
import pytest

from feature_selections.heuristics.population_based.genetic import Genetic
from feature_selections.heuristics.population_based.pbil import Pbil
from feature_selections.heuristics.population_based.tide import Tide
from feature_selections.heuristics.single_solution.forward import ForwardSelection
from feature_selections.heuristics.single_solution.local import LocalSearch


def _assert_contains_feature(mask, selector, feature):
    idx = selector.cols.get_loc(feature)
    assert bool(np.array(mask, dtype=bool)[idx])


def _build_selector(selector_cls, *, dataset, pipeline_factory, tmp_path, warm, **kwargs):
    return selector_cls(
        name=f"warm_{selector_cls.__name__.lower()}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        Tmax=1,
        suffix="",
        cv=None,
        verbose=False,
        output=str(tmp_path),
        warm_start=warm,
        **kwargs,
    )


def test_warm_start_forward_selection(dataset, pipeline_factory, tmp_path):
    selector = _build_selector(
        ForwardSelection,
        dataset=dataset,
        pipeline_factory=pipeline_factory,
        tmp_path=tmp_path,
        warm=["f1"],
        N=5,
        Gmax=0,
        ratio=0.2,
        strat="sfs",
    )

    score, mask, subset, *_ = selector.start(pid=0)
    _assert_contains_feature(mask, selector, "f1")
    assert subset == ["f1"]
    expected = 1 - selector.ratio * (1 / selector.D)
    assert score == pytest.approx(expected)


def test_warm_start_local_search(dataset, pipeline_factory, tmp_path):
    selector = _build_selector(
        LocalSearch,
        dataset=dataset,
        pipeline_factory=pipeline_factory,
        tmp_path=tmp_path,
        warm=["f1"],
        N=4,
        Gmax=0,
        ratio=0.2,
        nb=1,
    )

    score, mask, subset, *_ = selector.start(pid=0)
    _assert_contains_feature(mask, selector, "f1")
    assert "f1" in subset
    expected = 1 - selector.ratio * (1 / selector.D)
    assert score >= expected


@pytest.mark.parametrize(
    "selector_cls, params",
    [
        (Genetic, {"N": 4, "Gmax": 0, "ratio": 0.2, "mutation": 1, "entropy": 0.01}),
        (Tide, {"N": 4, "Gmax": 0, "ratio": 0.2, "gamma": 0.5, "filter_init": False, "sfs_init": False, "entropy": 0.01}),
    ],
)
def test_warm_start_population(dataset, pipeline_factory, tmp_path, selector_cls, params):
    selector = _build_selector(
        selector_cls,
        dataset=dataset,
        pipeline_factory=pipeline_factory,
        tmp_path=tmp_path,
        warm=["f1"],
        **params,
    )

    score, mask, subset, *_ = selector.start(pid=0)
    _assert_contains_feature(mask, selector, "f1")
    assert "f1" in subset
    expected = 1 - selector.ratio * (1 / selector.D)
    assert score >= expected


def test_warm_start_pbil(dataset, pipeline_factory, tmp_path):
    selector = _build_selector(
        Pbil,
        dataset=dataset,
        pipeline_factory=pipeline_factory,
        tmp_path=tmp_path,
        warm=["f1"],
        N=6,
        Gmax=1,
        ratio=0.2,
        LR=0.1,
        MP=0.0,
        MS=0.0,
        entropy=0.01,
        warm_start_prob=0.9,
        warm_start_cold_prob=0.1,
    )

    score, mask, subset, *_ = selector.start(pid=0)
    _assert_contains_feature(mask, selector, "f1")
    assert "f1" in subset
    expected = 1 - selector.ratio * (1 / selector.D)
    assert score >= expected
