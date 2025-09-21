import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "f1": [0, 0, 1, 1, 0, 0, 1, 1],
            "f2": [0, 1, 0, 1, 1, 0, 1, 0],
            "f3": [1, 1, 0, 0, 1, 1, 0, 0],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )


@pytest.fixture
def pipeline_factory():
    def _factory():
        return Pipeline(
            [
                ("model", LogisticRegression(solver="liblinear", random_state=0)),
            ]
        )

    return _factory
