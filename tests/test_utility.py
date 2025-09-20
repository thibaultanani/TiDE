"""Tests for utility helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility import utility


def test_read_prefers_repo_datasets_when_parent_missing_file(tmp_path, monkeypatch):
    """Ensure ``read`` falls back to the repository datasets directory."""

    project_dir = tmp_path / "repo"
    parent_dir = tmp_path

    repo_datasets = project_dir / "datasets"
    parent_datasets = parent_dir / "datasets"

    repo_datasets.mkdir(parents=True)
    parent_datasets.mkdir(parents=True)

    data = pd.DataFrame({"value": [1, 2, 3]})
    data.to_csv(repo_datasets / "example.csv", index=False)

    monkeypatch.chdir(project_dir)

    loaded = utility.read("example")

    pdt.assert_frame_equal(loaded.reset_index(drop=True), data)

    # Sanity check: no file exists in the parent datasets directory
    assert not Path(parent_datasets / "example.csv").exists()
