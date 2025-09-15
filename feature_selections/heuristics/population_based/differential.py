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
        strat (str)    : Mutation strategy :
                         "rand/1", "best/1", "current-to-rand/1", "current-to-best/1", "rand-to-best/1",
                         "rand/2", "best/2", "current-to-rand/2", "current-to-best/2", "rand-to-best/2".
                         default : "rand/1".
        entropy (float): Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, F=1.0, CR=0.5, strat=None, entropy=0.05, suffix=None, cv=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.F = F
        self.CR = CR
        self.strat = strat or "rand/1"
        valid_strategies = {
            "rand/1", "best/1", "current-to-rand/1", "current-to-best/1", "rand-to-best/1",
            "rand/2", "best/2", "current-to-rand/2", "current-to-best/2", "rand-to-best/2"
        }
        if self.strat not in valid_strategies:
            raise ValueError(f"strat '{self.strat}' invalid. Choose,  {valid_strategies}.")
        self.entropy = entropy
        self.indMax = None
        self.path = os.path.join(self.path, 'differential' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def mutate_rand(P, n_ind, F, current):
        """
        DE/rand/1 : M_i = S_{r1} + F * (S_{r2} - S_{r3})
        """
        candidates = [i for i in range(len(P)) if i != current]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        Xr1 = P[r1].astype(int)
        Xr2 = P[r2].astype(int)
        Xr3 = P[r3].astype(int)
        # Calcul bit-per-bit
        mutant = [1 if (Xr1[i] + F * (Xr2[i] - Xr3[i])) >= 0.5 else 0
                  for i in range(n_ind)]
        return mutant

    @staticmethod
    def mutate_best(P, n_ind, F, current, best):
        """
        DE/best/1 : M_i = S_{b} + F * (S_{r1} - S_{r2})
        """
        candidates = [i for i in range(len(P)) if i != current and i != best]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        Xb = P[best].astype(int)
        Xr1 = P[r1].astype(int)
        Xr2 = P[r2].astype(int)
        mutant = [1 if (Xb[i] + F * (Xr1[i] - Xr2[i])) >= 0.5 else 0
                  for i in range(n_ind)]
        return mutant

    @staticmethod
    def mutate_current_to_rand_1(P, n_ind, F, current):
        """
        DE/current-to-rand/1 : M_i = S_i + F * (S_{r1} - S_i) + F * (S_{r2} - S_{r3})
        """
        candidates = [i for i in range(len(P)) if i != current]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        Si = P[current].astype(int)
        Sr1 = P[r1].astype(int)
        Sr2 = P[r2].astype(int)
        Sr3 = P[r3].astype(int)
        mutant = [
            1 if (Si[i] + F * (Sr1[i] - Si[i]) + F * (Sr2[i] - Sr3[i])) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_current_to_best_1(P, n_ind, F, current, best):
        """
        DE/current-to-best/1 : M_i = S_i + F * (S_{b} - S_i) + F * (S_{r1} - S_{r2})
        """
        candidates = [i for i in range(len(P)) if i != current and i != best]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        Si = P[current].astype(int)
        Sb = P[best].astype(int)
        Sr1 = P[r1].astype(int)
        Sr2 = P[r2].astype(int)
        mutant = [
            1 if (Si[i] + F * (Sb[i] - Si[i]) + F * (Sr1[i] - Sr2[i])) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_rand_to_best_1(P, n_ind, F, current, best):
        """
        DE/rand-to-best/1 : M_i = S_{r1} + F * (S_{b} - S_i) + F * (S_{r2} - S_{r3})
        """
        candidates = [i for i in range(len(P)) if i != current]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        Xr1 = P[r1].astype(int)
        Sb = P[best].astype(int)
        Si = P[current].astype(int)
        Xr2 = P[r2].astype(int)
        Xr3 = P[r3].astype(int)
        mutant = [
            1 if (Xr1[i] + F * (Sb[i] - Si[i]) + F * (Xr2[i] - Xr3[i])) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_rand_2(P, n_ind, F, current):
        """
        DE/rand/2 : M_i = S_{r1} + F * (S_{r2} - S_{r3}) + F * (S_{r4} - S_{r5})
        """
        candidates = [i for i in range(len(P)) if i != current]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        Sr1 = P[r1].astype(int)
        Sr2 = P[r2].astype(int)
        Sr3 = P[r3].astype(int)
        Sr4 = P[r4].astype(int)
        Sr5 = P[r5].astype(int)
        mutant = [
            1 if (Sr1[i] + F * (Sr2[i] - Sr3[i]) + F * (Sr4[i] - Sr5[i])) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_best_2(P, n_ind, F, current, best):
        """
        DE/best/2 : M_i = S_{b} + F * (S_{r1} - S_{r2}) + F * (S_{r3} - S_{r4})
        """
        candidates = [i for i in range(len(P)) if i != current and i != best]
        r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
        Sb = P[best].astype(int)
        Sr1 = P[r1].astype(int)
        Sr2 = P[r2].astype(int)
        Sr3 = P[r3].astype(int)
        Sr4 = P[r4].astype(int)
        mutant = [
            1 if (Sb[i] + F * (Sr1[i] - Sr2[i]) + F * (Sr3[i] - Sr4[i])) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_current_to_rand_2(P, n_ind, F, current):
        """
        DE/current-to-rand/2 : M_i = S_i + F * (S_{r1} - S_i) + F * (S_{r2} - S_{r3}) + F * (S_{r4} - S_{r5})
        """
        candidates = [i for i in range(len(P)) if i != current]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        Si = P[current].astype(int)
        Sr1 = P[r1].astype(int)
        Sr2 = P[r2].astype(int)
        Sr3 = P[r3].astype(int)
        Sr4 = P[r4].astype(int)
        Sr5 = P[r5].astype(int)
        mutant = [
            1 if (
                         Si[i]
                         + F * (Sr1[i] - Si[i])
                         + F * (Sr2[i] - Sr3[i])
                         + F * (Sr4[i] - Sr5[i])
                 ) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_current_to_best_2(P, n_ind, F, current, best):
        """
        DE/current-to-best/2 : M_i = S_i + F * (S_{b} - S_i) + F * (S_{r1} - S_{r2}) + F * (S_{r3} - S_{r4})
        """
        candidates = [i for i in range(len(P)) if i != current and i != best]
        r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
        Si = P[current].astype(int)
        Sb = P[best].astype(int)
        Sr1 = P[r1].astype(int)
        Sr2 = P[r2].astype(int)
        Sr3 = P[r3].astype(int)
        Sr4 = P[r4].astype(int)
        mutant = [
            1 if (
                         Si[i]
                         + F * (Sb[i] - Si[i])
                         + F * (Sr1[i] - Sr2[i])
                         + F * (Sr3[i] - Sr4[i])
                 ) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def mutate_rand_to_best_2(P, n_ind, F, current, best):
        """
        DE/rand-to-best/2 : M_i = S_{r1} + F * (S_{b} - S_i) + F * (S_{r2} - S_{r3}) + F * (S_{r4} - S_{r5})
        """
        candidates = [i for i in range(len(P)) if i != current]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        Sr1 = P[r1].astype(int)
        Sb = P[best].astype(int)
        Si = P[current].astype(int)
        Sr2 = P[r2].astype(int)
        Sr3 = P[r3].astype(int)
        Sr4 = P[r4].astype(int)
        Sr5 = P[r5].astype(int)
        mutant = [
            1 if (
                         Sr1[i]
                         + F * (Sb[i] - Si[i])
                         + F * (Sr2[i] - Sr3[i])
                         + F * (Sr4[i] - Sr5[i])
                 ) >= 0.5 else 0
            for i in range(n_ind)
        ]
        return mutant

    @staticmethod
    def crossover(n_ind, ind, mutant, cross_proba):
        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    def specifics(self, bestInd, g, t, last, out):
        string = ("F factor: " + str(self.F) + os.linesep + "Crossover rate: " + str(self.CR) + os.linesep +
                  "Mutation strategy: (DE/" + str(self.strat) + ")" + os.linesep)
        name_strat = f"Differential Evolution (DE/{self.strat})"
        self.save(name_strat, bestInd, g, t, last, string, out)

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
                          pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0] for ind in P]
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
                best_idx = np.argmax(scores)
                if self.strat == "rand/1":
                    Vi = self.mutate_rand(P=P, n_ind=self.D, F=self.F, current=i)
                elif self.strat == "best/1":
                    Vi = self.mutate_best(P=P, n_ind=self.D, F=self.F, current=i, best=best_idx)
                elif self.strat == "current-to-rand/1":
                    Vi = self.mutate_current_to_rand_1(P=P, n_ind=self.D, F=self.F, current=i)
                elif self.strat == "current-to-best/1":
                    Vi = self.mutate_current_to_best_1(P=P, n_ind=self.D, F=self.F, current=i, best=best_idx)
                elif self.strat == "rand-to-best/1":
                    Vi = self.mutate_rand_to_best_1(P=P, n_ind=self.D, F=self.F, current=i, best=best_idx)
                elif self.strat == "rand/2":
                    Vi = self.mutate_rand_2(P=P, n_ind=self.D, F=self.F, current=i)
                elif self.strat == "best/2":
                    Vi = self.mutate_best_2(P=P, n_ind=self.D, F=self.F, current=i, best=best_idx)
                elif self.strat == "current-to-rand/2":
                    Vi = self.mutate_current_to_rand_2(P=P, n_ind=self.D, F=self.F, current=i)
                elif self.strat == "current-to-best/2":
                    Vi = self.mutate_current_to_best_2(P=P, n_ind=self.D, F=self.F, current=i, best=best_idx)
                elif self.strat == "rand-to-best/2":
                    Vi = self.mutate_rand_to_best_2(P=P, n_ind=self.D, F=self.F, current=i, best=best_idx)
                else:
                    raise ValueError(f"Unknown strategy : {self.strat}")
                # Child vector calculation Ui
                Ui = self.crossover(n_ind=self.D, ind=P[i], mutant=Vi, cross_proba=self.CR)
                # Evaluation of the trial vector
                if all(x == y for x, y in zip(P[i], Ui)):
                    score_ = scores[i]
                else:
                    score_ = fitness(train=self.train, test=self.test, columns=self.cols, ind=Ui, target=self.target,
                                     pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]
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
                          pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]for ind in P]
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
        return scoreMax, indMax, subsetMax, self.pipeline, pid, code, G - same2, G
