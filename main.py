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