import math
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from feature_selections.filters import Filter
from feature_selections.heuristics.population_based import Genetic, Differential, Pbil, Tide
from feature_selections.heuristics.single_solution import ForwardSelection, BackwardSelection, LocalSearch, Tabu
from feature_selections.heuristics import Random

from utility.utility import read
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge, LinearRegression, SGDClassifier, Lasso
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import rbf_kernel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

if __name__ == '__main__':
    datasets = ['A_baseline', 'A_instance', 'A_imbalanced', 'A_noise']
    stds = [True, True, True, True]
    drops_lists = [[], [], [], [],]
    metric = balanced_accuracy_score
    tmax = 3600
    k = 5
    verbose = True
    suffixes = ['_log', '_tree', '_knn']
    methods = [LogisticRegression(random_state=42, solver="lbfgs", class_weight="balanced", max_iter=10000),
               DecisionTreeClassifier(random_state=42, class_weight="balanced"),
               KNeighborsClassifier(weights='distance', algorithm='kd_tree')]
    tabus = []
    for (m, suffix) in zip(methods, suffixes):
        for i in range(len(datasets)):
            train = read(filename=datasets[i])
            try:
                m.n_neighbors = int(math.sqrt(train.shape[0]/k))
            except AttributeError:
                pass
            test = None
            name = datasets[i] + suffix
            target = "target"
            std = stds[i]
            drops = drops_lists[i]
            anov = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Anova")
            mrmr = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="MRMR")
            surf = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="SURF")
            rand = Random(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose)
            sffs = ForwardSelection(name=name, target=target, train=train, test=test, k=k, standardisation=std,
                                    drops=drops, metric=metric, model=m, Tmax=tmax, verbose=verbose)
            sfbs = BackwardSelection(name=name, target=target, train=train, test=test, k=k, standardisation=std,
                                     drops=drops, metric=metric, model=m, Tmax=tmax, verbose=verbose)
            local = LocalSearch(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                                metric=metric, model=m, Tmax=tmax, verbose=verbose)
            tabu = Tabu(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, verbose=verbose)
            gene = Genetic(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                           metric=metric, model=m, Tmax=tmax, verbose=verbose)
            pbil = Pbil(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, verbose=verbose)
            diff = Differential(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                                metric=metric, model=m, Tmax=tmax, suffix='_rand', verbose=verbose)
            diff2 = Differential(name=name, target=target, train=train, test=test, k=k, standardisation=std,
                                 drops=drops,  metric=metric, model=m, Tmax=tmax, suffix='_best', verbose=verbose,
                                 strat=True)
            tide = Tide(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, suffix='_anova', verbose=verbose)
            tide2 = Tide(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                         metric=metric, model=m, Tmax=tmax, verbose=verbose, filter_init=False)
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
                results_str += (f"Rang: {i}, method: {result[5]}, pid: {result[4]}, score: {round(result[0], 4)},"
                                f" classifier: {result[3].__class__.__name__}, n selected: {len(result[2])},"
                                f" convergence: {result[6]}, n iter: {result[7]},"
                                f" subset: {result[2]}\n")
            print(results_str)
            res_path = os.path.join(os.getcwd(), os.path.join(os.path.join('out', name), 'res.txt'))
            with open(res_path, "w") as file:
                file.write(results_str)
