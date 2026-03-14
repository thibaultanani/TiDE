"""Regression tests for AUAC/time-to-best reporting."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from feature_selections.filters.filter import Filter
from feature_selections.heuristics.population_based.tide import Tide
from feature_selections.heuristics.single_solution.forward import ForwardSelection
from helper.helper import fitness


@pytest.mark.parametrize(
    ("selector_cls", "params"),
    [
        (Filter, {"method": "correlation", "Gmax": 3, "Tmax": 1}),
        (ForwardSelection, {"strat": "sfs", "Gmax": 3, "Tmax": 1, "N": 4}),
    ],
)
def test_time_to_best_starts_with_full_feature_score(selector_cls, params, dataset, pipeline_factory, tmp_path):
    selector = selector_cls(
        name=f"time_to_best_{selector_cls.__name__.lower()}_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        ratio=0.1,
        verbose=False,
        output=str(tmp_path),
        **params,
    )

    selector.start(pid=3)

    full_score = float(
        fitness(
            train=selector.train,
            test=selector.test,
            columns=selector.cols,
            ind=np.ones(selector.D, dtype=bool),
            target=selector.target,
            pipeline=selector.pipeline,
            scoring=selector.scoring,
            ratio=selector.ratio,
            cv=selector.cv,
            rng=selector.rng,
        )[0]
    )

    curve_path = selector.path / "time_to_best_curve.json"
    log_path = selector.path / "log.txt"

    curve_points = json.loads(curve_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8")

    assert curve_points
    assert curve_points[0]["t"] == pytest.approx(0.0)
    assert curve_points[0]["p"] == selector.D
    assert curve_points[0]["score"] == pytest.approx(full_score)
    assert curve_points[-1]["t"] == pytest.approx(1.0)
    assert "AUAC (time-normalized):" in log_text
    assert "Time-to-best points:" in log_text


def test_tide_initialisations_consume_global_generations_and_are_logged(dataset, pipeline_factory, tmp_path):
    selector = Tide(
        name=f"time_to_best_tide_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        ratio=0.1,
        verbose=False,
        output=str(tmp_path),
        Gmax=2,
        Tmax=1,
        N=4,
        gamma=0.5,
        filter_init=True,
        sfs_init=True,
        entropy=0.01,
    )

    result = selector.start(pid=9)
    log_text = (selector.path / "log.txt").read_text(encoding="utf-8")

    assert result[-1] == 2
    assert "G:     1" in log_text
    assert "G:     2" in log_text


@pytest.mark.parametrize(
    ("selector_cls", "params", "expected_type"),
    [
        (Filter, {"method": "correlation", "Gmax": 3, "Tmax": 1}, "filter"),
        (ForwardSelection, {"strat": "sfs", "Gmax": 3, "Tmax": 1, "N": 4}, "heuristic"),
    ],
)
def test_results_json_is_written(selector_cls, params, expected_type, dataset, pipeline_factory, tmp_path):
    selector = selector_cls(
        name=f"results_json_{selector_cls.__name__.lower()}_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        ratio=0.1,
        verbose=False,
        output=str(tmp_path),
        **params,
    )

    selector.start(pid=5)
    results_payload = json.loads((selector.path / "results.json").read_text(encoding="utf-8"))

    assert results_payload["strategy_type"] == expected_type
    assert isinstance(results_payload["ba_cv"], float)
    assert results_payload["internal_evaluation_mode"] in {"cv", "holdout"}
    assert isinstance(results_payload["time_budget_seconds"], float)
    assert results_payload["method_mode"]
    assert results_payload["time_to_best_points_file"] == "time_to_best_curve.json"
    assert results_payload["time_to_best_points"]


def test_filter_cv_majority_mode_is_explicit_and_writes_results(dataset, pipeline_factory, tmp_path):
    selector = Filter(
        name=f"filter_cv_majority_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=None,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        ratio=0.1,
        verbose=False,
        output=str(tmp_path),
        method="correlation",
        selection_mode="cv_majority",
        Gmax=6,
        Tmax=1,
    )

    result = selector.start(pid=11)
    results_payload = json.loads((selector.path / "results.json").read_text(encoding="utf-8"))

    assert result[-1] <= selector.Gmax
    assert results_payload["selection_mode"] == "cv_majority"
    assert results_payload["selection_mode_label"] == "per-fold ranking + majority vote"
    assert isinstance(results_payload["best_subset"], list)
