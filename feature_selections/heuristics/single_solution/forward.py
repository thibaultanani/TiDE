import os
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, fitness


class ForwardSelection(Heuristic):
    """
    Class that implements the Stepwise Floating Forward Selection (SFFS) heuristic.
    """

    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, suffix=None, cv=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.selected_features = []  # Initialize with an empty set of selected features
        self.path = os.path.join(self.path, 'forward_selection' + self.suffix)
        createDirectory(path=self.path)

    def specifics(self, bestInd, g, t, last, out):
        self.save("Stepwise Floating Forward Selection", bestInd, g, t, last, "", out)

    @staticmethod
    def forward_backward_search(train, test, cols, D, target, pipeline, scoring, ratio, cv, selected_features,
                                scoreMax, indMax):
        improvement = False
        best_feature_to_add = None
        for feature in cols:
            if feature not in selected_features:
                candidate_features = selected_features + [feature]
                candidate = np.zeros(D, dtype=int)
                for var in candidate_features:
                    candidate[cols.get_loc(var)] = 1
                score = fitness(train=train, test=test, columns=cols, ind=candidate, target=target, pipeline=pipeline,
                                scoring=scoring, ratio=ratio, cv=cv)[0]
                if score > scoreMax:
                    scoreMax, indMax = score, candidate
                    best_feature_to_add = feature
                    improvement = True
        if best_feature_to_add is not None:
            selected_features.append(best_feature_to_add)
        best_feature_to_remove = None
        for feature in selected_features:
            candidate_features = [f for f in selected_features if f != feature]
            candidate = np.zeros(D, dtype=int)
            for var in candidate_features:
                if var != feature:
                    candidate[cols.get_loc(var)] = 1
            score = fitness(train=train, test=test, columns=cols, ind=candidate, target=target, pipeline=pipeline,
                            scoring=scoring, ratio=ratio, cv=cv)[0]
            if score > scoreMax:
                scoreMax, indMax = score, candidate
                best_feature_to_remove = feature
                improvement = True
        if best_feature_to_remove is not None:
            selected_features.remove(best_feature_to_remove)
        return improvement, selected_features, scoreMax, indMax

    def start(self, pid):
        code = "SFFS"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, indMax = -np.inf, 0
        # Generation (G) initialisation
        G, same_since_improv, stop = 0, 0, False
        # Main process iteration (generation iteration)
        improvement = True
        while G < self.Gmax and improvement:
            instant = time.time()
            improvement, self.selected_features, scoreMax, indMax = ForwardSelection.forward_backward_search(
                train=self.train, test=self.test, cols=self.cols, D=self.D, target=self.target, pipeline=self.pipeline,
                scoring=self.scoring, ratio=self.ratio, cv=self.cv, selected_features=self.selected_features,
                scoreMax=scoreMax, indMax=indMax)
            G = G + 1
            if improvement:
                same_since_improv = 0
            else:
                same_since_improv += 1
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=scoreMax,
                                     mean=scoreMax, feats=len(self.selected_features), time_exe=time_instant,
                                     time_total=time_debut, g=G, cpt=same_since_improv, verbose=self.verbose) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop or same_since_improv == self.D:
                self.specifics(bestInd=indMax, g=G, t=timedelta(seconds=(time.time() - debut)),
                               last=G - same_since_improv, out=print_out)
                print_out = ""
                if stop:
                    break
        return scoreMax, indMax, self.selected_features, self.pipeline, pid, code, G - same_since_improv, G
