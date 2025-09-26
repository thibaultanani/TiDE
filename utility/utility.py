"""Utility helpers shared across the TiDE feature selection library."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from numpy.random import Generator
import pandas as pd
from sklearn.model_selection import cross_val_predict


DatasetPath = Path | str


def _dataset_root(filename: str | None = None) -> Path:
    """Return the absolute path to the datasets directory.

    When both ``../datasets`` and ``./datasets`` exist, prefer the one that
    actually contains ``filename``; otherwise fall back to the closest existing
    directory to preserve backwards compatibility with older entrypoints.
    """

    cwd = Path(os.getcwd())
    parent = cwd.parent / "datasets"
    direct = cwd / "datasets"

    if filename is not None:
        for root in (parent, direct):
            if not root.exists():
                continue
            base = root / filename
            if base.with_suffix(".xlsx").exists() or base.with_suffix(".csv").exists():
                return root

    for root in (parent, direct):
        if root.exists():
            return root

    return direct


def _resolve_dataset_path(filename: str) -> Path:
    """Return the dataset path (without extension) for ``filename``.

    Parameters
    ----------
    filename:
        Base filename without extension.
    """

    return _dataset_root(filename) / filename


def read(filename: str, separator: str = ",") -> pd.DataFrame:
    """Load a dataset from either the CSV or XLSX representation.

    Parameters
    ----------
    filename:
        Base filename (without extension).
    separator:
        Column separator used in the CSV representation of the dataset.
    """

    path = _resolve_dataset_path(filename)
    try:
        return pd.read_excel(f"{path}.xlsx", index_col=None, engine="openpyxl")
    except FileNotFoundError:
        return pd.read_csv(f"{path}.csv", index_col=None, sep=separator)


def write(filename: str, data: pd.DataFrame) -> pd.DataFrame:
    """Persist a dataset to both CSV and XLSX when possible.

    Parameters
    ----------
    filename:
        Base filename (without extension).
    data:
        Data to serialise.
    """

    path = _resolve_dataset_path(filename)
    try:
        data.to_excel(f"{path}.xlsx", index=False)
    except FileNotFoundError:
        data.to_csv(f"{path}.csv", index=False)
    return data


def create_directory(path: DatasetPath) -> None:
    """Create a clean directory at ``path``.

    The target directory is **always deleted** before being recreated, so avoid
    pointing it at locations that may contain artefacts you need to keep.
    Existing experiment outputs are removed to avoid mixing results from
    different runs.  The legacy camel-case name is kept to preserve backwards
    compatibility with the public API, even though the implementation follows
    ``snake_case`` conventions internally.
    """

    final_path = Path(path)
    if final_path.exists():
        shutil.rmtree(final_path)
    final_path.mkdir(parents=True, exist_ok=True)


def create_population(inds: int, size: int, rng: Generator | None = None) -> np.ndarray:
    """Return a boolean population matrix initialised at random."""

    rng = rng or np.random.default_rng()
    thresholds = rng.random((inds, 1))
    pop = rng.random((inds, size)) < thresholds
    order = np.argsort(-rng.random(size), axis=0)
    pop = pop[:, order]
    return pop.astype(bool)


def _ensure_valid_individual(individual: Sequence[bool], rng: Generator | None = None) -> np.ndarray:
    """Ensure that at least one feature is selected in ``individual``."""

    as_array = np.asarray(individual, dtype=bool)
    if not as_array.any():
        as_array = as_array.copy()
        rng = rng or np.random.default_rng()
        random_index = int(rng.integers(0, len(as_array)))
        as_array[random_index] = True
    return as_array


def _prepare_subset(columns: Sequence[str], selected: Sequence[bool]) -> List[str]:
    """Return the column names flagged as ``True`` in ``selected``."""

    return [columns[idx] for idx, keep in enumerate(selected) if keep]


def fitness(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    columns: Sequence[str],
    ind: Sequence[bool],
    target: str,
    pipeline,
    scoring,
    ratio: float,
    cv=None,
    rng: Generator | None = None,
):
    """Evaluate an individual and return the penalised score and predictions.

    Returns
    -------
    tuple
        ``(score, y_true, y_pred)`` where ``score`` is penalised by the
        ``ratio`` hyper-parameter.
    """

    selected = _ensure_valid_individual(ind, rng=rng)
    subset = _prepare_subset(columns, selected)
    train_sub = train[subset + [target]]
    X_train, y_train = train_sub.drop(columns=[target]), train_sub[target]

    if test is None and cv is not None:
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=1)
        score = scoring(y_train, y_pred) - (ratio * (len(subset) / len(columns)))
        return score, y_train.reset_index(drop=True), pd.Series(y_pred)

    if test is None:
        raise ValueError("Either 'test' or 'cv' must be provided to compute fitness.")

    test_sub = test[subset + [target]]
    X_test, y_test = test_sub.drop(columns=[target]), test_sub[target]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = scoring(y_test, y_pred) - (ratio * (len(subset) / len(columns)))
    return score, y_test.reset_index(drop=True), pd.Series(y_pred)


def random_int_power(n: int, power: int = 2, rng: Generator | None = None) -> int:
    """Return an integer sampled from ``[1, n]`` with a power-law distribution."""

    rng = rng or np.random.default_rng()
    weights = np.array([1 / (i**power) for i in range(1, n + 1)])
    weights = weights / weights.sum()
    choices = np.arange(1, n + 1)
    return int(rng.choice(choices, p=weights))


def diversification(individual: Sequence[int], distance: int, rng: Generator | None = None) -> List[int]:
    """Create a neighbour by flipping ``distance`` random positions."""

    rng = rng or np.random.default_rng()
    neighbor = list(individual)
    size = len(neighbor)
    if distance >= 0:
        num_moves = int(rng.integers(1, max(1, distance) + 1))
    else:
        num_moves = random_int_power(n=size, power=2, rng=rng)
    if num_moves == 0:
        return neighbor
    move_indices = rng.choice(size, size=num_moves, replace=False)
    for idx in move_indices:
        neighbor[idx] = 1 - neighbor[idx]
    return neighbor


def get_entropy(pop: Sequence[Sequence[bool]]) -> float:
    """Compute the mean Shannon entropy of a population."""

    array_pop = np.asarray(pop, dtype=bool)
    if array_pop.size == 0:
        return 0.0
    probs = array_pop.mean(axis=0)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    entropy = -(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
    return float(np.mean(entropy))


def add(scores: Sequence[float], inds: Sequence[Sequence[bool]], cols: Sequence[str]) -> tuple[float, List[str], np.ndarray]:
    """Return the best score, subset of features and individual."""

    scores_array = np.asarray(scores)
    inds_array = np.asarray(inds)
    argmax = int(np.argmax(scores_array))
    best_score = float(scores_array[argmax])
    best_ind = inds_array[argmax]
    best_subset = _prepare_subset(cols, best_ind)
    return best_score, best_subset, best_ind
