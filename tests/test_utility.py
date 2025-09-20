from pathlib import Path

import pandas as pd
import pytest

from utility.utility import read, write


def test_read_uses_repo_dataset_when_sibling_missing(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    dataset_name = "internet"
    dataset_csv = repo_root / "datasets" / f"{dataset_name}.csv"
    if not dataset_csv.exists():
        pytest.skip("internet.csv dataset is required for this test")

    sibling_dir = repo_root.parent / "datasets"
    created_dir = False
    if not sibling_dir.exists():
        sibling_dir.mkdir()
        created_dir = True
    else:
        for extension in (".xlsx", ".csv"):
            if (sibling_dir / f"{dataset_name}{extension}").exists():
                pytest.skip("Sibling datasets directory already contains the target dataset")

    try:
        monkeypatch.chdir(repo_root)
        data = read(dataset_name)
        expected = pd.read_csv(dataset_csv, sep=",")
        pd.testing.assert_frame_equal(data, expected)
    finally:
        if created_dir:
            sibling_dir.rmdir()


def test_write_uses_repo_dataset_when_sibling_missing(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    dataset_name = "test_tmp_dataset"
    repo_dataset_dir = repo_root / "datasets"
    repo_files = [repo_dataset_dir / f"{dataset_name}{ext}" for ext in (".xlsx", ".csv")]
    sibling_dir = repo_root.parent / "datasets"
    sibling_files = [sibling_dir / f"{dataset_name}{ext}" for ext in (".xlsx", ".csv")]

    if any(path.exists() for path in repo_files + sibling_files):
        pytest.skip("Temporary dataset files already exist")

    created_dir = False
    if not sibling_dir.exists():
        sibling_dir.mkdir()
        created_dir = True

    data = pd.DataFrame({"value": [1, 2, 3]})

    try:
        monkeypatch.chdir(repo_root)
        write(dataset_name, data)

        repo_written_files = [path for path in repo_files if path.exists()]
        assert repo_written_files, "Dataset should be written into the repository datasets directory"
        assert not any(path.exists() for path in sibling_files), (
            "Dataset should not be written into the sibling datasets directory"
        )

        loaded = read(dataset_name)
        expected = data.reset_index(drop=True)
        pd.testing.assert_frame_equal(loaded.reset_index(drop=True), expected)
    finally:
        for path in repo_files + sibling_files:
            if path.exists():
                path.unlink()
        if created_dir:
            sibling_dir.rmdir()
