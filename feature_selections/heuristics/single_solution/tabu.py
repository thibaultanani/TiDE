import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from queue import Queue
from utility.utility import createDirectory, add, create_population_models, fitness


class Tabu(Heuristic):
    """
    Class that implements the tabu search heuristic.

    Args:
        size (int): Size of the tabu list
        nb (int)  : Maximal neighbors distance from the current solution
    """
    def __init__(self, name, target, train, test, model, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, size=None, nb=None, suffix=None, k=None, standardisation=None, verbose=None):
        super().__init__(name, target, model, train, test, k, standardisation, drops, metric, N, Gmax, Tmax, ratio,
                         suffix, verbose)
        self.size = size or self.N
        self.nb = nb or 5
        self.path = os.path.join(self.path, 'tabu' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def diversification(individual, distance, models):
        neighbor = individual.copy()
        bits_to_flip = random.sample(range(len(individual)), random.randint(0, distance))
        for chromosome in bits_to_flip:
            if chromosome != len(individual) - 1:
                neighbor[chromosome] = int(not neighbor[chromosome])
            else:
                r = random.randint(0, len(models) - 1)
                while r == neighbor[chromosome] and len(models) > 1:
                    r = random.randint(0, len(models) - 1)
                neighbor[chromosome] = r
        return neighbor

    @staticmethod
    def insert_tabu(tabuList, individual):
        if not tabuList.full():
            tabuList.put(individual)
        else:
            tabuList.get()
            tabuList.put(individual)
        return tabuList

    @staticmethod
    def add_neighborhood(scores, inds, cols):
        argmax = np.argmax(scores)
        bestScore = scores[argmax]
        bestInd = inds[argmax]
        bestSubset = [cols[i] for i in range(len(cols)) if bestInd[i]]
        return bestScore, bestSubset, bestInd, argmax

    @staticmethod
    def remove_neighbor(scores, inds, argmax):
        scores.pop(argmax)
        np.delete(inds, argmax)
        return scores, inds

    def specifics(self, bestInd, g, t, last, out):
        string = "Tabu List Size: " + str(self.size) + os.linesep + \
                 "Disruption (1 to Max): " + str(self.nb) + os.linesep
        self.save("Tabu Search", bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "TABU"
        debut = time.time()
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        # Measuring the execution time
        instant = time.time()
        # Generation (G) and Tabu List initialisation
        G, tabuList, same1, same2, stop = 0, Queue(maxsize=self.size), 0, 0, False
        # Population P initialisation
        P = create_population_models(inds=self.N, size=self.D + 1, models=self.model)
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
        tabuList = self.insert_tabu(tabuList=tabuList, individual=bestInd)
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            instant = time.time()
            # Neighborhood exploration and evaluation
            neighborhood = []
            for i in range(self.N):
                # Neighbor calculation
                neighbor = self.diversification(individual=bestInd, distance=self.nb, models=self.model)
                neighborhood.append(neighbor)
            # Evaluate the neighborhood
            scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                              models=self.model, metric=self.metric, standardisation=self.standardisation,
                              ratio=self.ratio, k=self.k)[0] for ind in neighborhood]
            bestScore, bestSubset, bestInd, argmax =\
                self.add_neighborhood(scores=scores, inds=np.asarray(neighborhood), cols=self.cols)
            while np.any(np.all(bestInd == list(tabuList.queue), axis=1)):
                scores, neighborhood = self.remove_neighbor(scores=scores, inds=neighborhood, argmax=argmax)
                bestScore, bestSubset, bestInd, argmax = \
                    self.add_neighborhood(scores=scores, inds=np.asarray(neighborhood), cols=self.cols)
            tabuList = self.insert_tabu(tabuList=tabuList, individual=bestInd)
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
            # If convergence is reached restart
            if same1 >= 300:
                same1 = 0
                P = create_population_models(inds=self.N, size=self.D + 1, models=self.model)
                # Evaluates population
                scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                                  models=self.model, metric=self.metric, standardisation=self.standardisation,
                                  ratio=self.ratio, k=self.k)[0] for ind in P]
                bestScore, bestSubset, bestInd = add(scores=scores, inds=np.asarray(P), cols=self.cols)
                tabuList = Queue(maxsize=self.size)
                tabuList = self.insert_tabu(tabuList=tabuList, individual=bestInd)
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
        return scoreMax, indMax, subsetMax, self.model[indMax[-1]], pid, code
