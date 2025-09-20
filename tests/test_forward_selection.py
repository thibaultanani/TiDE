import time
import uuid

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from feature_selections.heuristics.population_based.tide import Tide
from feature_selections.heuristics.single_solution.forward import ForwardSelection


@pytest.fixture
def dataset():
    data = pd.DataFrame(
        {
            "f1": [0, 0, 1, 1, 0, 0, 1, 1],
            "f2": [0, 1, 0, 1, 1, 0, 1, 0],
            "f3": [1, 1, 0, 0, 1, 1, 0, 0],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    return data


def _make_pipeline():
    return Pipeline(
        [
            ("model", LogisticRegression(solver="liblinear", random_state=0)),
        ]
    )


@pytest.fixture
def make_forward_selection(tmp_path, dataset):
    def _factory():
        return ForwardSelection(
            name=f"fs_test_{uuid.uuid4().hex}",
            target="target",
            pipeline=_make_pipeline(),
            train=dataset,
            test=dataset,
            Tmax=10,
            ratio=0.2,
            N=5,
            Gmax=5,
            suffix="",
            cv=None,
            verbose=False,
            output=str(tmp_path),
            strat="sfs",
        )

    return _factory


def _expected_indicator(num_features, index):
    indicator = np.zeros(num_features, dtype=int)
    indicator[index] = 1
    return indicator


def test_build_indicator_matches_selected_features(make_forward_selection):
    fs = make_forward_selection()
    indicator = fs._build_indicator(["f1", "f3"])
    expected = np.array([1, 0, 1])
    assert np.array_equal(indicator, expected)


def test_evaluate_features_returns_indicator_and_score(make_forward_selection):
    fs = make_forward_selection()
    indicator, score = fs._evaluate_features(["f1"])
    expected = _expected_indicator(fs.D, 0)
    assert np.array_equal(indicator, expected)
    expected_score = 1 - fs.ratio * (1 / fs.D)
    assert score == pytest.approx(expected_score)


def test_forward_step_adds_best_feature(make_forward_selection):
    fs = make_forward_selection()
    fs.selected_features = []
    score_max = float("-inf")
    ind_max = np.zeros(fs.D, dtype=int)

    improvement, selected, new_score, new_indicator, timeout = fs._forward_step(
        scoreMax=score_max,
        indMax=ind_max,
        start_time=time.time(),
    )

    assert improvement is True
    assert timeout is False
    assert selected == ["f1"]
    expected_indicator = _expected_indicator(fs.D, 0)
    assert np.array_equal(new_indicator, expected_indicator)
    expected_score = 1 - fs.ratio * (1 / fs.D)
    assert new_score == pytest.approx(expected_score)


def test_backward_step_removes_penalised_feature(make_forward_selection):
    fs = make_forward_selection()
    fs.selected_features = ["f1", "f2"]
    indicator_two, score_two = fs._evaluate_features(fs.selected_features)

    improvement, selected, new_score, new_indicator, timeout = fs._backward_step(
        scoreMax=score_two,
        indMax=indicator_two,
        start_time=time.time(),
    )

    assert improvement is True
    assert timeout is False
    assert selected == ["f1"]
    expected_indicator = _expected_indicator(fs.D, 0)
    assert np.array_equal(new_indicator, expected_indicator)
    expected_score = 1 - fs.ratio * (1 / fs.D)
    assert new_score == pytest.approx(expected_score)


def test_forward_backward_step_prefers_single_feature(make_forward_selection):
    fs = make_forward_selection()
    fs.selected_features = ["f1", "f2"]
    indicator_two, score_two = fs._evaluate_features(fs.selected_features)

    improvement, selected, new_score, new_indicator, timeout = fs._forward_backward_step(
        scoreMax=score_two,
        indMax=indicator_two,
        start_time=time.time(),
    )

    assert improvement is True
    assert timeout is False
    assert selected == ["f1"]
    expected_indicator = _expected_indicator(fs.D, 0)
    assert np.array_equal(new_indicator, expected_indicator)
    expected_score = 1 - fs.ratio * (1 / fs.D)
    assert new_score == pytest.approx(expected_score)


@pytest.mark.parametrize("sffs_init", [False, True])
def test_tide_forward_initialisation_returns_best_indicator(tmp_path, dataset, sffs_init):
    tide = Tide(
        name=f"tide_test_{uuid.uuid4().hex}",
        target="target",
        pipeline=_make_pipeline(),
        train=dataset,
        test=dataset,
        scoring=None,
        Tmax=10,
        ratio=0.2,
        N=5,
        Gmax=5,
        gamma=0.8,
        filter_init=False,
        sffs_init=sffs_init,
        entropy=0.05,
        suffix="",
        cv=None,
        verbose=False,
        output=str(tmp_path),
    )

    indicator = tide.forward_initialisation()
    expected_indicator = _expected_indicator(tide.D, 0)
    assert np.array_equal(indicator, expected_indicator)
    assert np.issubdtype(indicator.dtype, np.integer)
