import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, get_entropy, create_population, fitness, random_int_power


class Genetic(Heuristic):
    """
    Class that implements the genetic algorithm heuristic.

    Args:
        mutation (int): Maximum number of mutations for each child
        entropy (float) : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, mutation=None, entropy=None, suffix=None, cv=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix,
                         verbose, output)
        self.mutation = mutation if mutation is not None else -1
        self.entropy = entropy if entropy is not None else 0.05
        self.path = os.path.join(self.path, 'genetic' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def get_ranks(scores):
        ranks = {}
        for i, score in enumerate(sorted(scores)):
            rank = i + 1
            if score not in ranks:
                ranks[score] = rank
        return [ranks[score] for score in scores]

    def rank_selection(self, population, scores, n_select):
        ranks = self.get_ranks(scores)
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        selected_indices = np.random.choice(len(population), n_select, p=probabilities, replace=False)
        return [population[i] for i in selected_indices], [scores[i] for i in selected_indices]

    @staticmethod
    def roulette_selection(population, scores):
        min_score = min(scores)
        if min_score < 0:
            shifted_scores = [s - min_score + 1e-8 for s in scores]
        else:
            shifted_scores = scores
        total_score = sum(shifted_scores)
        if total_score == 0:
            probabilities = [1 / len(scores)] * len(scores)
        else:
            probabilities = [s / total_score for s in shifted_scores]
        selected_indices = np.random.choice(len(population), 2, p=probabilities, replace=False)
        return population[selected_indices[0]], population[selected_indices[1]]

    @staticmethod
    def crossover(parent1, parent2):
        parent1 = parent1.tolist()
        parent2 = parent2.tolist()
        if random.random() < 0.5:
            p1, p2 = parent2, parent1
        else:
            p1, p2 = parent1, parent2
        point1 = random.randint(0, len(p1) - 1)
        point2 = random.randint(point1, len(p1) - 1)
        child = p1[:point1] + p2[point1:point2] + p1[point2:]
        return child

    @staticmethod
    def mutate(individual, mutation):
        mutant = individual.copy()
        size = len(mutant)
        if mutation >= 0:
            num_moves = random.randint(0, mutation)
        else:
            num_moves = random_int_power(n=size + 1, power=2) - 1
        move_indices = random.sample(range(size), num_moves)
        for idx in move_indices:
            mutant[idx] = 1 - mutant[idx]
        return mutant

    @staticmethod
    def print_(print_out, pid, maxi, best, mean, worst, feats, time_exe, time_total, entropy, g, cpt):
        display = "[GENE]    PID: [{:3}]    G: {:5d}    max: {:2.4%}    features: {:6d}    best: {:2.4%}" \
                  "   mean: {:2.4%}    worst: {:2.4%}    G time: {}    T time: {}    last: {:6d}    entropy : {:2.3%}" \
            .format(pid, g, maxi, feats, best, mean, worst, time_exe, time_total, cpt, entropy)
        print_out = print_out + display
        print(display)
        return print_out

    def specifics(self, bestInd, g, t, last, out):
        string = "Mutation Rate: " + str(self.mutation) + os.linesep
        self.save("Genetic Algorithm", bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "GENE"
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
            # Selection of individuals for the next generation by rank
            P, scores = self.rank_selection(population=P, scores=scores, n_select=self.N)
            # Children population
            for _ in range(self.N):
                parent1, parent2 = self.roulette_selection(population=P, scores=scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(individual=child, mutation=self.mutation)
                child_score = fitness(train=self.train, test=self.test, columns=self.cols, ind=child,
                                      target=self.target, pipeline=self.pipeline, scoring=self.scoring,
                                      ratio=self.ratio, cv=self.cv)[0]
                scores.append(child_score)
                P.append(np.asarray(child))
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
