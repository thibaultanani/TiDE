import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from copy import copy
from utility.utility import createDirectory, add, get_entropy, fitness


class Pbil(Heuristic):
    """
    Class that implements the population based incremental learning heuristic.

    Args:
        LR (float)      : Learning rate, the speed at which the proba vector will converge
        MP (float)      : Mutation propability on a value in the proba vector
        MS (float)      : Mutation shift, the speed at which the proba vector will converge when a mutation occurs
        n (int)         : Number of individuals in the population used to update the proba vector
        entropy (float) : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, LR=0.1, MP=0.05, MS=0.1, n=None, entropy=0.05, suffix=None, cv=None, verbose=None,
                 output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.LR = LR
        self.MP = MP
        self.MS = MS
        self.n = n if n is not None else int(self.N * 0.15)
        self.entropy = entropy
        self.path = os.path.join(self.path, 'pbil' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def create_probas(size):
        probas = []
        for i in range(size):
            probas.append(0.5)
        return probas

    @staticmethod
    def top_n_average(P, scores, n):
        top_n_indices = np.argsort(scores)[-n:]
        top_n_vectors = [P[i] for i in top_n_indices]
        average_vector = np.mean(top_n_vectors, axis=0)
        return average_vector

    @staticmethod
    def update_probas(probas, LR, bests):
        for i in range(len(probas)):
            probas[i] = (probas[i] * (1.0 - LR)) + (LR * bests[i])
        return probas

    @staticmethod
    def mutate_probas(probas, MP, MS):
        for i in range(len(probas)):
            if random.random() < MP:
                probas[i] = (probas[i] * (1.0 - MS)) + (MS * random.choice([1.0, 0.0]))
        return probas

    @staticmethod
    def create_population(inds, size, probas):
        pop = np.zeros((inds, size), dtype=int)
        for i in range(inds):
            for j in range(len(pop[i])):
                if random.random() < probas[j]:
                    pop[i][j] = 1
                else:
                    pop[i][j] = 0
        return pop

    def specifics(self, probas, bestInd, g, t, last, out):
        string = "Learning Rate: " + str(self.LR) + os.linesep + \
                 "Mutation Probabilities: " + str(self.MP) + os.linesep + \
                 "Mutation Shift: " + str(self.MS) + os.linesep + \
                 "Probabilities Vector: " + str(['%.3f' % x for x in probas]) + os.linesep
        self.save("Population Based Incremental Learning", bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "PBIL"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, subsetMax, indMax = -np.inf, 0, 0
        # Generation (G) initialisation
        G, same1, same2, time_debut, stop = 0, 0, 0, 0, False
        # Initialize probabilities vector (0.5 to select each features)
        probas = self.create_probas(size=self.D)
        saves_proba = copy(probas)
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            # Measuring the execution time
            instant = time.time()
            P = self.create_population(inds=self.N, size=self.D, probas=probas)
            # Evaluates population
            scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                              pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0] for ind in P]
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
                saves_proba = copy(probas)
            print_out = self.pprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, feats=len(subsetMax), time_exe=time_instant,
                                     time_total=time_debut, entropy=entropy, g=G, cpt=same2, verbose=self.verbose) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                self.specifics(probas=saves_proba, bestInd=indMax, g=G,
                               t=timedelta(seconds=(time.time() - debut)), last=G - same2, out=print_out)
                print_out = ""
                if stop:
                    break
            # Update probabilies vectors with the best individuals of the generation
            bests = self.top_n_average(P=P, scores=scores, n=self.n)
            probas = self.update_probas(probas=probas, LR=self.LR, bests=bests)
            # Mutations to increase diversity and premature convergence
            probas = self.mutate_probas(probas=probas, MP=self.MP, MS=self.MS)
            # If diversity is too low restart
            if entropy < self.entropy:
                same1 = 0
                probas = self.create_probas(size=self.D)
        return scoreMax, indMax, subsetMax, self.pipeline, pid, code, G - same2, G
