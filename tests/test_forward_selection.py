import time
import uuid

import numpy as np
import pytest

from feature_selections.heuristics.population_based.tide import Tide
from feature_selections.heuristics.single_solution.forward import ForwardSelection


@pytest.fixture
def make_forward_selection(tmp_path, dataset, pipeline_factory):
    def _factory():
        return ForwardSelection(
            name=f"fs_test_{uuid.uuid4().hex}",
            target="target",
            pipeline=pipeline_factory(),
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


def test_evaluate_candidate_builds_expected_indicator(make_forward_selection):
    fs = make_forward_selection()
    score, indicator = fs._evaluate_candidate(["f1", "f3"])
    expected = np.array([1, 0, 1])
    assert np.array_equal(indicator, expected)
    expected_score = 1 - fs.ratio * (2 / fs.D)
    assert score == pytest.approx(expected_score)


def test_evaluate_candidate_returns_indicator_and_score(make_forward_selection):
    fs = make_forward_selection()
    score, indicator = fs._evaluate_candidate(["f1"])
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
    score_two, indicator_two = fs._evaluate_candidate(fs.selected_features)

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


def test_forward_step_rejects_penalised_feature(make_forward_selection):
    fs = make_forward_selection()
    fs.selected_features = ["f1"]
    score_one, indicator_one = fs._evaluate_candidate(fs.selected_features)

    improvement, selected, new_score, new_indicator, timeout = fs._forward_step(
        scoreMax=score_one,
        indMax=indicator_one,
        start_time=time.time(),
    )

    assert improvement is False
    assert timeout is False
    assert selected == ["f1"]
    assert np.array_equal(new_indicator, indicator_one)
    assert new_score == pytest.approx(score_one)


def test_start_sffs_prefers_single_feature(tmp_path, dataset, pipeline_factory):
    selector = ForwardSelection(
        name=f"fs_sffs_test_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
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
        strat="sffs",
    )

    score, indicator, selected, *_ = selector.start(pid=0)

    expected_indicator = _expected_indicator(selector.D, 0)
    assert np.array_equal(indicator, expected_indicator)
    assert selected == ["f1"]
    expected_score = 1 - selector.ratio * (1 / selector.D)
    assert score == pytest.approx(expected_score)


@pytest.mark.parametrize("sfs_init", [False, True])
def test_tide_forward_initialisation_returns_best_indicator(tmp_path, dataset, pipeline_factory, sfs_init):
    tide = Tide(
        name=f"tide_test_{uuid.uuid4().hex}",
        target="target",
        pipeline=pipeline_factory(),
        train=dataset,
        test=dataset,
        scoring=None,
        Tmax=10,
        ratio=0.2,
        N=5,
        Gmax=5,
        gamma=0.8,
        filter_init=False,
        sfs_init=sfs_init,
        entropy=0.05,
        suffix="",
        cv=None,
        verbose=False,
        output=str(tmp_path),
    )

    indicator = np.asarray(tide.forward_initialisation(), dtype=int)
    expected_indicator = _expected_indicator(tide.D, 0)
    assert np.array_equal(indicator, expected_indicator)
    assert np.issubdtype(indicator.dtype, np.integer)
