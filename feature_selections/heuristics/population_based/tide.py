import os
import random
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from skrebate import ReliefF
from utility.utility import createDirectory, add, get_entropy, create_population_models, fitness


class Tide(Heuristic):
    """
    Class that implements the new heuristic: tournament in differential evolution

    Args:
        alpha (float)     : The minimum threshold for the percentage of individuals to be selected by tournament
        Tmax_filter (int) : Total number of seconds allocated for searching best features subset with filter method
        filter_init (bool): The choice of using a filter method for initialisation
        entropy (float)   : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None, ratio=None,
                 N=None, Gmax=None, alpha=None, Tmax_filter=None, filter_init=None, entropy=None, suffix=None,
                 k=None, standardisation=None, verbose=None):
        super().__init__(name, target, model, train, test, k, standardisation, drops, metric, N, Gmax, Tmax, ratio,
                         suffix, verbose)
        self.alpha = alpha or 0.9
        self.Tmax_filter = Tmax_filter or Tmax / 10
        if filter_init is False:
            self.filter_init = False
        else:
            self.filter_init = True
        self.entropy = entropy or 0.02
        self.path = os.path.join(self.path, 'tide' + self.suffix)
        createDirectory(path=self.path)

    def rf_init(self):
        debut = time.time()
        cols = self.train.drop([self.target], axis=1).columns
        X = self.train.drop([self.target], axis=1).values.astype('float64')
        y = self.train[self.target].values
        relief = ReliefF(n_neighbors=100, n_features_to_select=X.shape[1])
        relief.fit(X, y)
        rel_scores = relief.feature_importances_
        rel_results = list(zip(cols, rel_scores))
        rel_results.sort(key=lambda x: x[1], reverse=True)
        sorted_features = [feature for feature, _ in rel_results]
        score, model, col, vector, G = -np.inf, 0, 0, 0, 0
        while G < self.Gmax:
            k = random.randint(1, self.D)
            top_k_features = sorted_features[:k]
            G = G + 1
            v = [0] * self.D
            for var in top_k_features:
                v[self.cols.get_loc(var)] = 1
            v.append(random.choice(range(len(self.model))))
            s = fitness(train=self.train, test=self.test, columns=self.cols, ind=v, target=self.target,
                        models=self.model, metric=self.metric, standardisation=self.standardisation,
                        ratio=self.ratio, k=self.k)[0]
            if s > score:
                score, vector = s, v
                col = [self.cols[i] for i in range(len(self.cols)) if v[i]]
            if time.time() - debut >= self.Tmax_filter:
                break
        if self.verbose:
            print("filter:", score, len(col), self.model[vector[-1]].__class__.__name__)
        return vector

    @staticmethod
    def mutate(P, n_ind, current, selected):
        list_of_index = [i for i in range(len(P))]
        r = np.random.choice(list_of_index, 2, replace=False)
        while current in r:
            r = np.random.choice(list_of_index, 2, replace=False)
        selected = [selected, r[0], r[1]]
        Xr1 = P[selected[0]]
        Xr2 = P[selected[1]]
        Xr3 = P[selected[2]]
        mutant = []
        for chromosome in range(n_ind):
            if Xr2[chromosome] == Xr3[chromosome]:
                mutant.append(Xr1[chromosome])
            else:
                mutant.append(Xr2[chromosome])
        return mutant

    @staticmethod
    def crossover(n_ind, ind, mutant, cross_proba):
        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    @staticmethod
    def tournament(scores, entropy, alpha):
        p = (1 - entropy) * (1 - alpha) + alpha
        nb_scores = int(scores.__len__() * p)
        selected = random.choices(scores, k=nb_scores)
        if len(selected) < 2:
            selected = random.choices(scores, k=2)
        score_max = np.amax(selected)
        for i, score in enumerate(scores):
            if score == score_max:
                return i

    def specifics(self, bestInd, g, t, last, out):
        if self.filter_init:
            string = "k: " + str(self.k)
            name = "Tournament In Differential Evolution + ReliefF"
        else:
            string = "k: No filter initialization"
            name = "Tournament In Differential Evolution"
        string = string + os.linesep + "Alpha: " + str(self.alpha) + os.linesep
        self.save(name, bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "TIDE"
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
        P = create_population_models(inds=self.N, size=self.D + 1, models=self.model)
        r = None
        if self.filter_init:
            r = self.rf_init()
            P[0] = r
        # Evaluates population
        scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                          models=self.model, metric=self.metric, standardisation=self.standardisation,
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
                # Calculate the rank for each individual
                selected = self.tournament(scores=scores, entropy=entropy, alpha=self.alpha)
                # Mutant calculation Vi
                Vi = self.mutate(P=P, n_ind=self.D + 1, current=i, selected=selected)
                # Child vector calculation Ui
                CR = random.uniform(0.3, 0.7)
                Ui = self.crossover(n_ind=self.D + 1, ind=P[i], mutant=Vi, cross_proba=CR)
                # Evaluation of the trial vector
                if all(x == y for x, y in zip(P[i], Ui)):
                    score_ = scores[i]
                else:
                    score_ = fitness(train=self.train, test=self.test, columns=self.cols, ind=Ui, target=self.target,
                                     models=self.model, metric=self.metric, standardisation=self.standardisation,
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
                P = create_population_models(inds=self.N, size=self.D + 1, models=self.model)
                if self.filter_init:
                    P[0] = r
                scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                                  models=self.model, metric=self.metric, standardisation=self.standardisation,
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
        return scoreMax, indMax, subsetMax, self.model[indMax[-1]], pid, code, G - same2, G
