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


def create_population_models(inds, size, models):
    pop = np.random.rand(inds, size) < np.random.rand(inds, 1)
    pop = pop[:, np.argsort(-np.random.rand(size), axis=0)]
    pop = pop.astype(int)
    for i in range(inds):
        pop[i, -1] = np.random.randint(0, len(models))
    return pop


def fitness(train, test, columns, ind, target, models, metric, standardisation, ratio, k=None):
    if not any(ind[:-1]):
        ind[random.randint(0, len(ind) - 2)] = 1
    subset = [columns[c] for c in range(len(columns)) if ind[c]]
    train = train[subset + [target]]
    X_train, y_train = train.drop(columns=[target]), train[target]
    model = models[ind[-1]]
    if test is None and k is not None:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = []
        all_y_val = []
        all_y_pred = []
        for train_index, val_index in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            if standardisation:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = metric(y_val, y_pred) - (ratio * (len(subset) / len(columns)))
            scores.append(score)
            all_y_val.extend(y_val)
            all_y_pred.extend(y_pred)
        avg_score = np.mean(scores)
        return avg_score, pd.Series(all_y_val), pd.Series(all_y_pred)
    else:
        test = test[subset + [target]]
        X_test, y_test = test.drop(columns=[target]), test[target]
        if standardisation:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred) - (ratio * (len(subset) / len(columns)))
        return score, y_test, y_pred



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


def get_entropy2(pop):
    H, ratio = [], []
    # Loop over the columns
    for i in range(len(pop[0]) - 1):
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
        ratio.append(p_true)
    return sum(H) / len(H), ratio


def add(scores, inds, cols):
    argmax = np.argmax(scores)
    bestScore = scores[argmax]
    bestInd = inds[argmax]
    bestSubset = [cols[i] for i in range(len(cols)) if bestInd[i]]
    return bestScore, bestSubset, bestInd
