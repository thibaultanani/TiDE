from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, multilabel_confusion_matrix


class FeatureSelection:
    """Common building blocks shared by every feature selection strategy."""

    def __init__(
        self,
        name: str,
        target: str,
        pipeline: Any,
        train: pd.DataFrame,
        test: Optional[pd.DataFrame] = None,
        cv: Any = None,
        drops: Optional[Sequence[str]] = None,
        scoring: Optional[Any] = None,
        Gmax: Optional[int] = None,
        Tmax: Optional[int] = None,
        ratio: Optional[float] = None,
        suffix: Optional[str] = None,
        verbose: Optional[bool] = None,
        output: Optional[str] = None,
        warm_start: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise a feature selection experiment.

        Parameters
        ----------
        name:
            Identifier used to create the results directory.
        target:
            Name of the prediction target present in ``train`` (and ``test``).
        pipeline:
            Scikit-learn compatible estimator (or pipeline) used for evaluation.
        train:
            Training dataset containing the target column.
        test:
            Optional hold-out dataset. If ``None``, a cross-validation strategy
            must be provided through ``cv``.
        cv:
            Cross-validation strategy used when ``test`` is ``None``.
        drops:
            Optional iterable of column names to remove from both ``train`` and
            ``test`` prior to optimisation.
        scoring:
            Scoring callable returning a higher value for better performance.
        Gmax:
            Maximum number of iterations/generations explored by the concrete
            strategy. ``None`` disables the iteration limit.
        Tmax:
            Time budget (in seconds). Once exceeded, optimisation stops
            gracefully. ``None`` means no explicit wall-clock budget.
        ratio:
            Penalty weight applied to the proportion of selected features when
            scoring subsets (``0`` disables the sparsity penalty).
        suffix:
            Optional string appended to output directories, useful to
            differentiate runs sharing the same ``name``.
        verbose:
            When ``True``, progress information is printed to stdout.
        output:
            Base directory where experiment artefacts (logs, models, reports)
            are written. Defaults to ``./out/<name>`` relative to the current
            working directory.
        warm_start:
            Optional iterable of feature names. For selectors supporting
            warm-starts, this subset seeds the initial individual (or
            probability vector for PBIL). Names not present in ``train`` are
            ignored.
        seed:
            Optional integer that seeds the pseudo-random number generator used
            by stochastic selectors.
        """

        drops = list(drops or [])
        self.train = train.drop(columns=drops, errors="ignore")
        self.test = test.drop(columns=drops, errors="ignore") if isinstance(test, pd.DataFrame) else None
        self.name = name
        self.target = target
        self.cols = self.train.drop([target], axis=1).columns
        self.n_class = int(self.train[target].nunique(dropna=False))
        self.D = len(self.cols)
        self.scoring = scoring or balanced_accuracy_score
        self.pipeline = pipeline
        self.cv = cv
        self.Gmax = Gmax or 1_000_000
        self.Tmax = Tmax or 3_600
        self.ratio = ratio if ratio is not None else 1e-5
        self.suffix = suffix or ""
        self.verbose = True if verbose is None else verbose
        base_path = Path(os.getcwd()) / (output or "out") / self.name
        base_path.mkdir(parents=True, exist_ok=True)
        self.base_path = base_path
        self.path: Path = base_path
        self._seed = seed if seed is None else int(seed)
        self._rng: Generator = default_rng(self._seed)

        warm_features = list(dict.fromkeys(col for col in (warm_start or []) if col in self.cols))
        mask = np.zeros(self.D, dtype=bool)
        for feature in warm_features:
            idx = self.cols.get_loc(feature)
            mask[idx] = True
        if not mask.any():
            self._warm_start_mask: Optional[np.ndarray] = None
            self.warm_start_features: list[str] = []
        else:
            self._warm_start_mask = mask
            self.warm_start_features = warm_features

    @abc.abstractmethod
    def start(self, pid: int) -> None:
        """Execute the feature selection strategy."""

    @staticmethod
    def calculate_confusion_matrix_components(y_true: pd.Series, y_pred: pd.Series) -> tuple[int, int, int, int]:
        """Return the confusion matrix components aggregated across classes."""

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return int(tp), int(tn), int(fp), int(fn)

        multilabel_cm = multilabel_confusion_matrix(y_true, y_pred)
        tp = int(multilabel_cm[:, 1, 1].sum())
        tn = int(multilabel_cm[:, 0, 0].sum())
        fp = int(multilabel_cm[:, 0, 1].sum())
        fn = int(multilabel_cm[:, 1, 0].sum())
        return tp, tn, fp, fn

    def reset_rng(self) -> None:
        """Reset the internal random generator to its initial seed."""

        self._rng = default_rng(self._seed)

    @property
    def rng(self) -> Generator:
        """Return the internal NumPy random generator."""

        return self._rng
