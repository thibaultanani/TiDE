import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, get_entropy, create_population, fitness


class Nmbde(Heuristic):
    """
    Class that implements the novel modified binary differential evolution heuristic.

    Args:
        Fmax (float)   : Maximum amplification factor for differential variation (default 0.8)
        Fmin (float)   : Minimum amplification factor for differential variation (default 0.005)
        CR (float)     : Crossover probability (default 0.2)
        strat (bool)   : Mutation strategy (False for DE/rand/1 or True for DE/best/1)
        b (float)      : Bandwidth factor for probability estimation operator (default 20)
        entropy (float): Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None,
                 N=None, Gmax=None, Fmax=0.8, Fmin=0.005, CR=0.2, strat=False, b=20.0, entropy=None, suffix=None,
                 cv=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        # Differential evolution parameters
        self.Fmax = Fmax
        self.Fmin = Fmin
        self.CR = CR
        self.strat = strat
        self.b = b
        self.entropy = entropy if entropy is not None else 0.05
        self.path = os.path.join(self.path, 'nmbde' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def _prob_est(MO, F, b):
        """
        Compute probability vector from MO using logistic function.
        """
        denom = 1.0 + 2.0 * F
        return 1.0 / (1.0 + np.exp(-2.0 * b * (MO - 0.5) / denom))

    @staticmethod
    def mutate_rand(P, n_ind, F, current, b):
        # select three distinct indices
        idx = [i for i in range(len(P)) if i != current]
        r1, r2, r3 = np.random.choice(idx, 3, replace=False)
        Xr1, Xr2, Xr3 = P[r1].astype(int), P[r2].astype(int), P[r3].astype(int)
        # differential estimate
        MO = Xr1 + F * (Xr2 - Xr3)
        # probability estimation
        P_vec = Nmbde._prob_est(MO, F, b)
        # sample mutant
        return [1 if random.random() < P_vec[i] else 0 for i in range(n_ind)]

    @staticmethod
    def mutate_best(P, n_ind, F, current, best, b):
        idx = [i for i in range(len(P)) if i not in (current, best)]
        r1, r2 = np.random.choice(idx, 2, replace=False)
        Xr1, Xr2, Xb = P[r1].astype(int), P[r2].astype(int), P[best].astype(int)
        # differential estimate
        MO = Xb + F * (Xr1 - Xr2)
        # probability estimation
        P_vec = Nmbde._prob_est(MO, F, b)
        # sample mutant
        return [1 if random.random() < P_vec[i] else 0 for i in range(n_ind)]

    @staticmethod
    def crossover(n_ind, ind, mutant, cross_proba):
        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        # ensure at least one gene from mutant
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child.tolist()

    def specifics(self, bestInd, g, t, last, out):
        string = (f"Fmax: {self.Fmax}\n"
                  f"Fmin: {self.Fmin}\n"
                  f"Crossover rate: {self.CR}\n"
                  f"Bandwidth b: {self.b}\n")
        label = "DE/best/1" if self.strat else "DE/rand/1"
        self.save(f"Novel Modified Binary Differential Evolution ({label})", bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "MBDE"
        debut = time.time()
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)

        # initial population
        G, same1, same2 = 0, 0, 0
        P = create_population(inds=self.N, size=self.D)
        # initial evaluation
        scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                          pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0] for ind in P]
        bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
        scoreMax, subsetMax, indMax = bestScore, bestSubset, bestInd

        # main loop
        while G < self.Gmax:
            # update F based on elapsed time
            elapsed = time.time() - debut
            frac = min(elapsed / self.Tmax, 1.0)
            F = self.Fmax - (self.Fmax - self.Fmin) * frac

            instant = time.time()
            for i in range(self.N):
                # choose mutation strategy with dynamic F
                if self.strat:
                    Vi = self.mutate_best(P, self.D, F, i, np.argmax(scores), self.b)
                else:
                    Vi = self.mutate_rand(P, self.D, F, i, self.b)
                Ui = np.array(self.crossover(self.D, P[i], Vi, self.CR), dtype=int)

                # evaluate
                if not np.array_equal(P[i], Ui):
                    score_i = fitness(train=self.train, test=self.test, columns=self.cols, ind=Ui, target=self.target,
                                      pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]
                else:
                    score_i = scores[i]

                # selection
                if score_i >= scores[i]:
                    P[i], scores[i] = Ui, score_i
                    bestScore, subsetMax, indMax = add(scores=scores, inds=np.asarray(P), cols=self.cols)

            G += 1; same1 += 1; same2 += 1
            mean_scores = float(np.mean(scores))
            entropy = get_entropy(pop=P)
            time_instant = timedelta(seconds=(time.time() - instant))
            time_total = timedelta(seconds=(time.time() - debut))

            if bestScore > scoreMax:
                same1, same2 = 0, 0
                scoreMax, subsetMax = bestScore, subsetMax

            print_out = self.pprint_(print_out=print_out, name=code, pid=pid,
                                     maxi=scoreMax, best=bestScore, mean=mean_scores,
                                     feats=len(subsetMax), time_exe=time_instant,
                                     time_total=time_total, entropy=entropy,
                                     g=G, cpt=same2, verbose=self.verbose) + "\n"

            # diversity reset
            if entropy < self.entropy:
                same1 = 0
                P = create_population(inds=self.N, size=self.D)
                P[0] = indMax
                scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                          pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]for ind in P]
                bestScore, subsetMax, indMax = add(scores=scores, inds=np.asarray(P), cols=self.cols)

            # periodic save
            if G % 10 == 0 or G == self.Gmax or (time.time() - debut) >= self.Tmax:
                self.specifics(bestInd=indMax, g=G,
                               t=timedelta(seconds=(time.time() - debut)),
                               last=G - same2, out=print_out)
                print_out = ""
            if (time.time() - debut) >= self.Tmax:
                break

        return scoreMax, indMax, subsetMax, self.pipeline, pid, code, G - same2, G
