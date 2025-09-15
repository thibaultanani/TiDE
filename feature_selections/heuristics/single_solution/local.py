import os
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, create_population, fitness, diversification


class LocalSearch(Heuristic):
    """
    Class that implements the local search heuristic.

    Args:
        nb (int)  : Maximal neighbors distance from the current solution
    """
    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, size=None, nb=None, suffix=None, cv=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.size = size if size is not None else self.N
        self.nb = nb if nb is not None else -1
        self.path = os.path.join(self.path, 'local' + self.suffix)
        createDirectory(path=self.path)

    def specifics(self, bestInd, g, t, last, out):
        string = "Disruption Rate: " + str(self.nb) + os.linesep
        self.save("Local Search", bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "LS  "
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        # Measuring the execution time
        instant = time.time()
        # Generation (G) initialisation
        G, same1, same2, stop = 0, 0, 0, False
        # Population P initialisation
        P = create_population(inds=self.N, size=self.D)
        # Evaluates population
        scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                          pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0] for ind in P]
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
        scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd
        mean_scores = float(np.mean(scores))
        time_instant = timedelta(seconds=(time.time() - instant))
        time_debut = timedelta(seconds=(time.time() - debut))
        print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                 mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                 time_total=time_debut, g=G, cpt=0, verbose=self.verbose) + "\n"
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            instant = time.time()
            # Neighborhood exploration and evaluation
            neighborhood = []
            for i in range(self.N):
                # Neighbor calculation
                neighbor = diversification(individual=indMax, distance=self.nb)
                neighborhood.append(neighbor)
            # Evaluate the neighborhood
            scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                              pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]
                      for ind in neighborhood]
            bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(neighborhood), cols=self.cols)
            G = G + 1
            same1, same2 = same1 + 1, same2 + 1
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            # Update which individual is the best
            if bestScore > scoreMax:
                same1, same2 = 0, 0
                scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                     time_total=time_debut, g=G, cpt=same2, verbose=self.verbose) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                self.specifics(bestInd=indMax, g=G, t=timedelta(seconds=(time.time() - debut)), last=G - same2,
                               out=print_out)
                print_out = ""
                if stop:
                    break
        return scoreMax, indMax, subsetMax, self.pipeline, pid, code, G - same2, G
