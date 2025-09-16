# TiDE: Tournament in Differential Evolution for Feature Selection

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://img.shields.io/pypi/v/tide-feature-selection.svg)](https://pypi.org/project/tide-feature-selection/)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-tide--feature--selection-informational)](https://test.pypi.org/project/tide-feature-selection/)

TiDE (Tournament in Differential Evolution) is a Python package that provides a comprehensive benchmark of filter-based, greedy, and metaheuristic wrapper methods for feature selection in machine learning. It also introduces a novel adaptive strategy based on Differential Evolution, capable of adjusting its mutation, initialization, and crossover policies according to the data.

This project was developed as part of a research study evaluating the robustness, performance, and extensibility of feature selection methods under various data conditions (noise, redundancy, imbalance, high-dimensionality). It is particularly suited for binary classification problems in high-dimensional settings.

## 🚀 Key Features

- A unified framework for evaluating feature selection methods.
- Integrated filter methods: ANOVA, MRMR, SURF.
- Greedy wrappers: Sequential Forward and Backward Floating Selection.
- Metaheuristics:
  - Local search: Hill Climbing, Tabu Search
  - Population-based: Genetic Algorithm, PBIL, DE, MBDE
- **TiDE**: A novel adaptive variant of Differential Evolution.


## 📦 Installation

You can install TiDE directly from PyPI:

```bash
pip install tide-feature-selection
```

Alternatively, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/thibaultanani/TiDE.git
cd TiDE
pip install -e .
```

Dependencies are listed in `setup.py` and will be automatically installed. These include:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `openpyxl`
- `psutil`

You can also create a virtual environment beforehand:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

## 🧪 Usage Example

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_selections.heuristics import Tide

if __name__ == '__main__':
    # Load the dataset
    data = load_breast_cancer()
    X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name='target')
    # Divide data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Divide the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Create DataFrames
    train_df, train_df['target'] = X_train.copy(), y_train
    val_df, val_df['target'] = X_val.copy(), y_val
    test_df, test_df['target'] = X_test.copy(), y_test
    # Create scikit-learn pipeline
    model = GaussianNB()
    scoring = balanced_accuracy_score
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    # Example of the use of a feature selection method
    tide = Tide(name="n1", target='target', train=train_df, test=val_df, scoring=scoring, pipeline=pipeline,
                Tmax=60, verbose=True, output="test")
    tide.start(pid=1)
    # It is also possible to only use training data as input for cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    tide_kfold = Tide(name="n2", target='target', train=train_df, cv=cv, scoring=scoring, pipeline=pipeline,
                      Tmax=60, verbose=True, output="test")
    tide_kfold.start(pid=2)
    # The results are automatically saved in the output "test" directory
```

## 🧠 Scientific Background

This package was developed as part of a research study investigating the robustness and adaptability of feature selection strategies across diverse data challenges. The proposed method TiDE dynamically adapts its mutation and crossover mechanisms according to the data characteristics, making it highly competitive compared to classical DE and other metaheuristics.

The full study is detailed in the accompanying manuscript:

> Anani, T., Delbot, F., & Pradat-Peyre, J.-F. (2025). *Tournament in Differential Evolution for Robust Feature Selection*. Currently being submitted.

## 📊 Citation

```bibtex
@article{anani2025tide,
  author = {Anani, Thibault and Delbot, François and Pradat-Peyre, Jean-François},
  title = {Tournament in Differential Evolution for Robust Feature Selection},
  journal = { },
  year = {2025},
  note = {Submitted}
}
```

## 🛠 Contributing

Contributions, ideas and bug reports are welcome! Please open an issue or a pull request.

## 📄 License

This project is licensed under the [BSD 3-Clause License](LICENSE).
