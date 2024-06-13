import os
import time
import numpy as np
import warnings

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, fitness

warnings.filterwarnings('ignore')


class Random(Heuristic):
    """
    Class that implements the random search heuristic
    """
    def __init__(self, name, target, train, test, model, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, suffix=None, k=None, standardisation=None, verbose=None):
        super().__init__(name, target, model, train, test, k, standardisation, drops, metric, N, Gmax, Tmax, ratio,
                         suffix, verbose)
        self.path = os.path.join(self.path, 'rand' + self.suffix)
        createDirectory(path=self.path)

    def specifics(self, bestInd, g, t, last, out):
        self.save("Random Generation", bestInd, g, t, last, "", out)

    @staticmethod
    def create_population_models(inds, size, models):
        # Initialise the population
        pop = np.zeros((inds, size), dtype=bool)
        for i in range(inds):
            num_true = np.random.randint(1, size)
            true_indices = np.random.choice(size - 1, size=num_true, replace=False)
            pop[i, true_indices] = True
        pop = pop.astype(int)
        # Replace last element with random integer between 0 and models-1
        pop[:, -1] = np.random.randint(0, len(models), size=inds)
        return pop

    def start(self, pid):
        code = "RAND"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        # Measuring the execution time
        instant = time.time()
        # Generation (G) initialisation
        G, same, stop = 0, 0, False
        # Population P initialisation
        P = self.create_population_models(inds=self.N, size=self.D + 1, models=self.model)
        # Evaluates population
        scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                          models=self.model, metric=self.metric, standardisation=self.standardisation,
                          ratio=self.ratio, k=self.k)[0] for ind in P]
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
        scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd
        mean_scores = float(np.mean(scores))
        time_instant = timedelta(seconds=(time.time() - instant))
        time_debut = timedelta(seconds=(time.time() - debut))
        # Pretty print the results
        print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                 mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                 time_total=time_debut, g=G, cpt=0, verbose=self.verbose) + "\n"
        scoreMax, subsetMax = bestScore, bestSubset
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            instant = time.time()
            # Neighborhood exploration and evaluation
            neighborhood = self.create_population_models(inds=self.N, size=self.D + 1, models=self.model)
            # Evaluate the neighborhood
            scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                              models=self.model, metric=self.metric, standardisation=self.standardisation,
                              ratio=self.ratio, k=self.k)[0] for ind in neighborhood]
            bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(neighborhood), cols=self.cols)
            bestInd = bestInd.tolist()
            G = G + 1
            same = same + 1
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            # Update which individual is the best
            if bestScore > scoreMax:
                same = 0
                scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                     time_total=time_debut, g=G, cpt=same, verbose=self.verbose) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                # Write important information to file
                self.specifics(bestInd=indMax, g=G, t=timedelta(seconds=(time.time() - debut)), last=G - same,
                               out=print_out)
                print_out = ""
                if stop:
                    break
        return scoreMax, indMax, subsetMax, self.model[indMax[-1]], pid, code

