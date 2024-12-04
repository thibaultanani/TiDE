import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, get_entropy, create_population, fitness


class Differential(Heuristic):
    """
    Class that implements the differential evolution heuristic.

    Args:
        F (float)      : Probability factor controlling the amplification of the differential variation
        CR (float)     : Crossover probability
        strat (bool)   : Mutation strategy (False for DE/rand/1 or True for DE/best/1)
        entropy (float): Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, F=None, CR=None, strat=None, entropy=None, suffix=None, k=None, standardisation=None,
                 verbose=None):
        super().__init__(name, target, model, train, test, k, standardisation, drops, metric, N, Gmax, Tmax, ratio,
                         suffix, verbose)
        self.F = F or 1.0
        self.CR = CR or 0.5
        self.strat = strat or False
        self.entropy = entropy or 0.05
        self.indMax = None
        self.path = os.path.join(self.path, 'differential' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def mutate_rand(P, n_ind, F, current):
        selected = np.random.choice([i for i in range(len(P)) if i != current], 3, replace=False)
        Xr1, Xr2, Xr3 = P[selected[0]], P[selected[1]], P[selected[2]]
        Xr1, Xr2, Xr3 = Xr1.astype(int), Xr2.astype(int), Xr3.astype(int)
        mutant = [1 if (Xr1[i] + F * (Xr2[i] - Xr3[i])) >= 0.5 else 0 for i in range(n_ind)]
        return mutant

    @staticmethod
    def mutate_best(P, n_ind, F, current, best):
        selected = np.random.choice([i for i in range(len(P)) if i != current and i != best], 2, replace=False)
        Xr1, Xr2, Xr3 = P[best], P[selected[0]], P[selected[1]]
        Xr1, Xr2, Xr3 = Xr1.astype(int), Xr2.astype(int), Xr3.astype(int)
        mutant = [1 if (Xr1[i] + F * (Xr2[i] - Xr3[i])) >= 0.5 else 0 for i in range(n_ind)]
        return mutant

    @staticmethod
    def crossover(n_ind, ind, mutant, cross_proba):
        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    def specifics(self, bestInd, g, t, last, out):
        string = "F factor: " + str(self.F) + os.linesep + "Crossover rate: " + str(self.CR) + os.linesep
        if self.strat:
            self.save("Differential Evolution (DE/best/1)", bestInd, g, t, last, string, out)
        else:
            self.save("Differential Evolution (DE/rand/1)", bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "DIFF"
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
                          model=self.model, metric=self.metric, standardisation=self.standardisation,
                          ratio=self.ratio, k=self.k)[0] for ind in P]
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
        scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd
        mean_scores = float(np.mean(scores))
        time_instant = timedelta(seconds=(time.time() - instant))
        time_debut = timedelta(seconds=(time.time() - debut))
        # Calculate diversity in population
        entropy = get_entropy(pop=P)
        # Pretty print the results
        print_out = self.pprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                 mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                 time_total=time_debut, entropy=entropy, g=G, cpt=0, verbose=self.verbose) + "\n"
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            instant = time.time()
            # Mutant population creation and evaluation
            for i in range(self.N):
                # Mutant calculation Vi
                if self.strat:
                    Vi = self.mutate_best(P=P, n_ind=self.D, F=self.F, current=i, best=np.argmax(scores))
                else:
                    Vi = self.mutate_rand(P=P, n_ind=self.D, F=self.F, current=i)
                # Child vector calculation Ui
                Ui = self.crossover(n_ind=self.D, ind=P[i], mutant=Vi, cross_proba=self.CR)
                # Evaluation of the trial vector
                if all(x == y for x, y in zip(P[i], Ui)):
                    score_ = scores[i]
                else:
                    score_ = fitness(train=self.train, test=self.test, columns=self.cols, ind=Ui, target=self.target,
                                     model=self.model, metric=self.metric, standardisation=self.standardisation,
                                     ratio=self.ratio, k=self.k)[0]
                # Comparison between Xi and Ui
                if scores[i] <= score_:
                    # Update population
                    P[i], scores[i] = Ui, score_
                    bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
            G = G + 1
            same1, same2 = same1 + 1, same2 + 1
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            entropy = get_entropy(pop=P)
            # Update which individual is the best
            if bestScore > scoreMax:
                same1, same2 = 0, 0
                scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd
            print_out = self.pprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                     time_total=time_debut, entropy=entropy, g=G, cpt=same2, verbose=self.verbose) + "\n"
            # If diversity is too low restart
            if entropy < self.entropy:
                same1 = 0
                P = create_population(inds=self.N, size=self.D)
                P[0] = indMax
                scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                                  model=self.model, metric=self.metric, standardisation=self.standardisation,
                                  ratio=self.ratio, k=self.k)[0] for ind in P]
                bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
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
        return scoreMax, indMax, subsetMax, self.model, pid, code, G - same2, G
