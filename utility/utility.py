import math
import os
import random
import shutil
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, \
    r2_score
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


def read(filename, separator=','):
    if os.path.exists(os.path.join(os.getcwd(), os.path.join('../datasets', filename))):
        path = os.path.join(os.getcwd(), os.path.join('../datasets', filename))
    else:
        path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


def write(filename, data):
    if os.path.exists(os.path.join(os.getcwd(), os.path.join('../datasets', filename))):
        path = os.path.join(os.getcwd(), os.path.join('../datasets', filename))
    else:
        path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data.to_excel(path + '.xlsx', index=False)
    except FileNotFoundError:
        data.to_csv(path + '.csv', index=False)
    return data


def createDirectory(path):
    # Clears the record of previous experiments on the same dataset with the same heuristic
    final = path
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(final)


def create_population(inds, size):
    # Initialise the population
    pop = np.random.rand(inds, size) < np.random.rand(inds, 1)
    pop = pop[:, np.argsort(-np.random.rand(size), axis=0)]
    return pop.astype(bool)


def fitness(train, test, columns, ind, target, pipeline, scoring, ratio, cv=None):
    """
    train : DataFrame (training)
    test : DataFrame (test) (None if Cross validation)
    columns : list of candidate column names
    ind : binary individual (feature selection)
    target : name of the target feature
    pipeline : scikit-learn pipeline including preprocessing + model
    scoring : sklearn score function (e.g. accuracy_score)
    ratio : penalty coefficient
    cv : cross-validation object (StratifiedKFold, KFold, LOO, etc.)
    """
    if not any(ind[:-1]):
        ind[random.randint(0, len(ind) - 1)] = 1
    subset = [columns[c] for c in range(len(columns)) if ind[c]]
    train_sub = train[subset + [target]]
    X_train, y_train = train_sub.drop(columns=[target]), train_sub[target]
    if test is None and cv is not None:
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=1)
        score = scoring(y_train, y_pred) - (ratio * (len(subset) / len(columns)))
        return score, y_train.reset_index(drop=True), pd.Series(y_pred)
    else:
        test_sub = test[subset + [target]]
        X_test, y_test = test_sub.drop(columns=[target]), test_sub[target]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = scoring(y_test, y_pred) - (ratio * (len(subset) / len(columns)))
        return score, y_test.reset_index(drop=True), pd.Series(y_pred)


def random_int_power(n, power=2):
    p = np.array([1 / (i ** power) for i in range(1, n + 1)])
    p = p / p.sum()
    return np.random.choice(range(1, n + 1), p=p)


def diversification(individual, distance):
    neighbor = individual.copy()
    size = len(neighbor)
    if distance >= 0:
        num_moves = random.randint(1, distance)
    else:
        num_moves = random_int_power(n=size, power=2)
    move_indices = random.sample(range(size), num_moves)
    for idx in move_indices:
        neighbor[idx] = 1 - neighbor[idx]
    return neighbor


def get_entropy(pop):
    H = []
    # Loop over the columns
    for i in range(len(pop[0])):
        # Initialize variables to store the counts of True and False values
        true_count = 0
        false_count = 0
        # Loop over the rows and count the number of True and False values in the current column
        for row in pop:
            if row[i]:
                true_count += 1
            else:
                false_count += 1
        # Calculate the probabilities of True and False values
        p_true = true_count / (true_count + false_count)
        p_false = false_count / (true_count + false_count)
        # Calculate the Shannon's entropy for the current column
        if p_true == 0 or p_false == 0:
            entropy = 0
        else:
            entropy = -(p_true * math.log2(p_true) + p_false * math.log2(p_false))
        # Append the result to the list
        H.append(entropy)
    return sum(H) / len(H)


def add(scores, inds, cols):
    argmax = np.argmax(scores)
    bestScore = scores[argmax]
    bestInd = inds[argmax]
    bestSubset = [cols[i] for i in range(len(cols)) if bestInd[i]]
    return bestScore, bestSubset, bestInd