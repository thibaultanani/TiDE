import math
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge, LinearRegression, SGDClassifier, Lasso
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from feature_selections.filters import Filter
from feature_selections.heuristics.population_based import Genetic, Differential, Pbil, Tide
from feature_selections.heuristics.single_solution import ForwardSelection, BackwardSelection, LocalSearch, Tabu
from feature_selections.heuristics import Random

from utility.utility import read

if __name__ == '__main__':
    datasets = ['A_baseline']
    stds = [True]
    drops_lists = [[]]
    scoring = balanced_accuracy_score
    tmax = 60 * 30
    cv = StratifiedKFold(random_state=42, shuffle=True, n_splits=5)
    verbose = True
    suffixes = ['_gnb']
    methods = [GaussianNB()]
    for (m, suffix) in zip(methods, suffixes):
        for i in range(len(datasets)):
            train = read(filename=datasets[i])
            test = None
            name = datasets[i] + suffix
            target = "target"
            std = stds[i]
            drops = drops_lists[i]
            if std:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', m)
                ])
            else:
                pipeline = Pipeline([
                    ('clf', m)
                ])
            anov = Filter(name=name, target=target, train=train, test=test, cv=cv,
                          drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                          verbose=verbose, method="Anova")
            mrmr = Filter(name=name, target=target, train=train, test=test, cv=cv,
                          drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                          verbose=verbose, method="MRMR")
            surf = Filter(name=name, target=target, train=train, test=test, cv=cv,
                          drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                          verbose=verbose, method="SURF")
            rand = Random(name=name, target=target, train=train, test=test, cv=cv,
                          drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                          verbose=verbose)
            sffs = ForwardSelection(name=name, target=target, train=train, test=test, cv=cv,
                                    drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                                    verbose=verbose)
            sfbs = BackwardSelection(name=name, target=target, train=train, test=test, cv=cv,
                                     drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                                     verbose=verbose)
            local = LocalSearch(name=name, target=target, train=train, test=test, cv=cv,
                                drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                                verbose=verbose)
            tabu = Tabu(name=name, target=target, train=train, test=test, cv=cv,
                        drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                        verbose=verbose)
            gene = Genetic(name=name, target=target, train=train, test=test, cv=cv,
                           drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                           verbose=verbose)
            pbil = Pbil(name=name, target=target, train=train, test=test, cv=cv,
                        drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                        verbose=verbose)
            diff = Differential(name=name, target=target, train=train, test=test, cv=cv,
                                drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                                strat='rand/1', verbose=verbose, suffix='_rand')
            diff2 = Differential(name=name, target=target, train=train, test=test, cv=cv,
                                 drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                                 strat='best/1', verbose=verbose, suffix='_best')
            tide = Tide(name=name, target=target, train=train, test=test, cv=cv,
                        drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                        suffix='_anova', verbose=verbose)
            tide2 = Tide(name=name, target=target, train=train, test=test, cv=cv,
                         drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                         verbose=verbose, filter_init=False)
            selection_methods = [anov, mrmr, surf, rand, sffs, sfbs, local, tabu, gene, pbil, diff, diff2, tide, tide2]
            num_processes = multiprocessing.cpu_count()
            results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {executor.submit(method.start, j + 1): j for j, method in enumerate(selection_methods)}
                for future in as_completed(futures):
                    id_method = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"An error has occurred with the method {id_method + 1}: {str(e)}")
                        traceback.print_exc()
            results = sorted(results, key=lambda x: x[0], reverse=True)
            results_str = "\nRankings:\n\n"
            i = 0
            for result in results:
                i = i + 1
                clf_name = result[3].steps[-1][1].__class__.__name__
                results_str += (f"Rang: {i}, method: {result[5]}, pid: {result[4]}, "
                                f"score: {round(result[0], 4)}, model: {clf_name}, "
                                f"n selected: {len(result[2])}, convergence: {result[6]}, "
                                f"n iter: {result[7]}, subset: {result[2]}\n")
            print(results_str)
            res_path = os.path.join(os.getcwd(), os.path.join(os.path.join('out', name), 'res.txt'))
            with open(res_path, "w") as file:
                file.write(results_str)
