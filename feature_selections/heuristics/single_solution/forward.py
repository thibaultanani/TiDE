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

    def __init__(self, name, target, pipeline, train, test, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, suffix=None, cv=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.selected_features = []  # Initialize with an empty set of selected features
        self.path = os.path.join(self.path, 'forward_selection' + self.suffix)
        createDirectory(path=self.path)

    def specifics(self, bestInd, g, t, last, out):
        self.save("Stepwise Floating Forward Selection", bestInd, g, t, last, "", out)

    def start(self, pid):
        code = "SFFS"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, indMax = -np.inf, 0
        # Generation (G) initialisation
        G, same1, same2, stop = 0, 0, 0, False
        remaining_features = list(range(self.D))
        # Main process iteration (generation iteration)
        improvement = True
        while G < self.Gmax and improvement:
            instant = time.time()
            improvement = False
            # Step 2: Forward step - add the best feature that improves fitness
            best_feature_to_add = None
            for feature in self.cols:
                if feature not in self.selected_features:
                    candidate_features = self.selected_features + [feature]
                    candidate = np.zeros(self.D, dtype=int)
                    for var in candidate_features:
                        candidate[self.cols.get_loc(var)] = 1
                    score = fitness(train=self.train, test=self.test, columns=self.cols, ind=candidate,
                                    target=self.target, pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio,
                                    cv=self.cv)[0]
                    if score > scoreMax:
                        scoreMax, indMax = score, candidate
                        best_feature_to_add = feature
                        improvement = True
            if best_feature_to_add is not None:
                self.selected_features.append(best_feature_to_add)
            # Step 3: Backward step - check if removing any feature improves the score
            best_feature_to_remove = None
            for feature in self.selected_features:
                candidate_features = [f for f in self.selected_features if f != feature]
                candidate = np.zeros(self.D, dtype=int)
                for var in candidate_features:
                    candidate[self.cols.get_loc(var)] = 1
                score = fitness(train=self.train, test=self.test, columns=self.cols, ind=candidate,
                                target=self.target, pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio,
                                cv=self.cv)[0]
                if score > scoreMax:
                    scoreMax, indMax = score, candidate
                    best_feature_to_remove = feature
                    improvement = True
            if best_feature_to_remove is not None:
                self.selected_features.remove(best_feature_to_remove)
            G = G + 1
            same1, same2 = same1 + 1, same2 + 1
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            # Update which individual is the best
            if improvement:
                same1, same2 = 0, 0
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=scoreMax,
                                     mean=scoreMax, feats=len(self.selected_features), time_exe=time_instant,
                                     time_total=time_debut, g=G, cpt=same2, verbose=self.verbose) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop or not remaining_features or not improvement:
                self.specifics(bestInd=indMax, g=G, t=timedelta(seconds=(time.time() - debut)), last=G - same2,
                               out=print_out)
                print_out = ""
                if stop or not remaining_features:
                    break
        return scoreMax, indMax, self.selected_features, self.pipeline, pid, code, G - same2, G
