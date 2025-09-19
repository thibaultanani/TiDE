"""Utility helpers shared across the TiDE feature selection library."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict


DatasetPath = Path | str


def _dataset_root() -> Path:
    """Return the absolute path to the datasets directory.

    The project historically stored datasets either in ``datasets`` or in
    ``../datasets`` depending on the execution entrypoint. This helper keeps the
    existing behaviour while offering a single place to update in the future.
    """

    cwd = Path(os.getcwd())
    direct = cwd / "datasets"
    parent = cwd.parent / "datasets"
    return parent if parent.exists() else direct


def _resolve_dataset_path(filename: str) -> Path:
    """Return the dataset path (without extension) for ``filename``.

    Parameters
    ----------
    filename:
        Base filename without extension.
    """

    return _dataset_root() / filename


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


def createDirectory(path: DatasetPath) -> None:  # noqa: N802 - legacy name kept for compatibility
    """Create a clean directory at ``path``.

    Existing experiment outputs are removed to avoid mixing results from
    different runs.  The legacy camel-case name is kept to preserve backwards
    compatibility with the public API, even though the implementation follows
    ``snake_case`` conventions internally.
    """

    final_path = Path(path)
    if final_path.exists():
        shutil.rmtree(final_path)
    final_path.mkdir(parents=True, exist_ok=True)


def create_population(inds: int, size: int) -> np.ndarray:
    """Return a boolean population matrix initialised at random."""

    pop = np.random.rand(inds, size) < np.random.rand(inds, 1)
    pop = pop[:, np.argsort(-np.random.rand(size), axis=0)]
    return pop.astype(bool)


def _ensure_valid_individual(individual: Sequence[bool]) -> np.ndarray:
    """Ensure that at least one feature is selected in ``individual``."""

    as_array = np.asarray(individual, dtype=bool)
    if not as_array.any():
        as_array = as_array.copy()
        random_index = random.randrange(len(as_array))
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
):
    """Evaluate an individual and return the penalised score and predictions.

    Returns
    -------
    tuple
        ``(score, y_true, y_pred)`` where ``score`` is penalised by the
        ``ratio`` hyper-parameter.
    """

    selected = _ensure_valid_individual(ind)
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


def random_int_power(n: int, power: int = 2) -> int:
    """Return an integer sampled from ``[1, n]`` with a power-law distribution."""

    weights = np.array([1 / (i**power) for i in range(1, n + 1)])
    weights = weights / weights.sum()
    return int(np.random.choice(range(1, n + 1), p=weights))


def diversification(individual: Sequence[int], distance: int) -> List[int]:
    """Create a neighbour by flipping ``distance`` random positions."""

    neighbor = list(individual)
    size = len(neighbor)
    if distance >= 0:
        num_moves = random.randint(1, max(1, distance))
    else:
        num_moves = random_int_power(n=size, power=2)
    move_indices = random.sample(range(size), num_moves)
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
