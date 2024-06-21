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
from feature_selections.heuristics.single_solution import LocalSearch, Tabu
from feature_selections.heuristics import Random

from utility.utility import read
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge, LinearRegression, SGDClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import rbf_kernel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

if __name__ == '__main__':
    datasets = ['baseline']
    stds = [True]
    metric = balanced_accuracy_score
    tmax = 3600*5
    k = 5
    verbose = True
    methods = [[LogisticRegression(random_state=42, solver="lbfgs", class_weight="balanced", max_iter=10000)],
               [RandomForestClassifier(n_estimators=5, class_weight="balanced", random_state=42)],
               [KNeighborsClassifier(weights='distance')],
               [SVC(random_state=42, class_weight="balanced", kernel=rbf_kernel)]]
    suffixes = ['logistic', 'random', 'knn', 'svc']
    for l in range(len(methods)):
        m = methods[l]
        suffix = suffixes[l]
        for i in range(len(datasets)):
            train = read(filename=datasets[i])
            test = None
            name = datasets[i] + suffix
            target = "target"
            std = stds[i]
            drops = []
            rand = Random(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose)
            local = LocalSearch(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                                metric=metric, model=m, Tmax=tmax, verbose=verbose)
            tabu = Tabu(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, verbose=verbose)
            gene = Genetic(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                           metric=metric, model=m, Tmax=tmax, verbose=verbose)
            diff = Differential(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                                metric=metric, model=m, Tmax=tmax, verbose=verbose)
            pbil = Pbil(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, verbose=verbose)
            tide = Tide(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, suffix='_randomforest', verbose=verbose)
            tide2 = Tide(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                         metric=metric, model=m, Tmax=tmax, verbose=verbose, filter_init=False)
            corr = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Correlation")
            anov = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Anova")
            info = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Mutual Information")
            mrmr = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="MRMR")
            reli = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                          metric=metric, model=m, Tmax=tmax, verbose=verbose, method="ReliefF")
            # surf = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
            #               metric=metric, model=m, Tmax=tmax, verbose=verbose, method="SURF")
            # multi = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
            #                metric=metric, model=m, Tmax=tmax, verbose=verbose, method="MultiSURF")
            # turf = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
            #               metric=metric, model=m, Tmax=tmax, verbose=verbose, method="TURF")
            # forest = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
            #                 metric=metric, model=m, Tmax=tmax, verbose=verbose, method="RandomForest")
            selection_methods = [rand, local, tabu, gene, diff, pbil, tide, tide2, corr, anov, info, mrmr, reli]
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
