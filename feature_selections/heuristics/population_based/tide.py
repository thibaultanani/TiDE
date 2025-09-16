import os
import random
import time
import numpy as np
from scipy.stats import beta
from sklearn.base import ClassifierMixin

from feature_selections.filters import Filter
from feature_selections.heuristics.single_solution import ForwardSelection
from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, get_entropy, create_population, fitness


class Tide(Heuristic):
    """
    Class that implements the new heuristic: tournament in differential evolution

    Args:
        gamma (float)       : The minimum threshold for the percentage of individuals to be selected by tournament
        filter_init (bool)  : The choice of using filter method for initialisation (anova or correlation learning task)
        sffs_init (bool)    : The choice of using sffs method for initialisation
        entropy (float)     : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, pipeline, train, test=None, drops=None, scoring=None, Tmax=None, ratio=None,
                 N=None, Gmax=None, gamma=None, filter_init=None, sffs_init=None, entropy=None, suffix=None, cv=None,
                 verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, N, Gmax, Tmax, ratio, suffix,
                         verbose, output)
        self.gamma = gamma if gamma is not None else 0.8
        if filter_init is False:
            self.filter_init, self.filter_str = False, ''
        else:
            self.filter_init = True
            if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
                self.filter_str = ' + ANOVA'
            else:
                self.filter_str = ' + CORRELATION'
        if sffs_init is False:
            self.sffs_init, self.sffs_str = False, ''
        else:
            self.sffs_init, self.sffs_str = True, ' + SFFS'
        self.entropy = entropy if entropy is not None else 0.05
        self.path = os.path.join(self.path, 'tide' + self.suffix)
        createDirectory(path=self.path)

    def filter_initialisation(self):
        debut = time.time()
        if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
            sorted_features, f_results = Filter.anova_selection(df=self.train, target=self.target)
        else:
            sorted_features, f_results = Filter.correlation_selection(df=self.train, target=self.target)
        score, model, col, vector, G = -np.inf, 0, 0, 0, 0
        k = [val for val in range(1, 101)]
        num_features = [int(round(len(sorted_features) * (val / 100.0))) for val in k]
        num_features = [val for val in num_features if val >= 1]
        num_features = list(dict.fromkeys(num_features))
        while G < self.Gmax:
            top_k_features = sorted_features[:num_features[G]]
            G = G + 1
            v = [0] * self.D
            for var in top_k_features:
                v[self.cols.get_loc(var)] = 1
            s = fitness(train=self.train, test=self.test, columns=self.cols, ind=v, target=self.target,
                        pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]
            if s > score:
                score, vector = s, v
            if time.time() - debut >= self.Tmax or G == len(num_features):
                break
        return vector

    def forward_initialisation(self):
        debut = time.time()
        selected_features = []
        scoreMax, indMax, G = -np.inf, 0, 0
        improvement = True
        while G < self.Gmax and improvement:
            improvement, selected_features, scoreMax, indMax = ForwardSelection.forward_step(
                train=self.train, test=self.test, cols=self.cols, D=self.D, target=self.target, pipeline=self.pipeline,
                scoring=self.scoring, ratio=self.ratio, cv=self.cv, selected_features=selected_features,
                scoreMax=scoreMax, indMax=indMax, start_time=debut, Tmax=self.Tmax)
            G = G + 1
            if time.time() - debut >= self.Tmax:
                break
        return indMax

    @staticmethod
    def mutate(P, n_ind, current, tbest):
        selected = np.random.choice([i for i in range(len(P)) if i != current and i != tbest], 2, replace=False)
        Xr1, Xr2, Xr3 = P[tbest], P[selected[0]], P[selected[1]]
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
    def tournament(scores, entropy, gamma):
        p = (1 - entropy) * (1 - gamma) + gamma
        nb_scores = max(2, int(len(scores) * p))
        selected = random.choices(scores, k=nb_scores)
        score_max = np.amax(selected)
        return scores.index(score_max)

    def specifics(self, bestInd, g, t, last, out):
        name = "Tournament In Differential" + self.filter_str + self.sffs_str
        string = "Gamma: " + str(self.gamma) + os.linesep
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
        P = create_population(inds=self.N, size=self.D)
        r1, r2 = None, None
        if self.filter_init:
            r1, r2 = self.filter_initialisation(), self.forward_initialisation()
            P[0], P[1] = r1, r2
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
                # Calculate the rank for each individual
                tbest = self.tournament(scores=scores, entropy=entropy, gamma=self.gamma)
                # Mutant calculation Vi
                Vi = self.mutate(P=P, n_ind=self.D, current=i, tbest=tbest)
                # Child vector calculation Ui
                score_i = max(scores[i], 0)
                alpha, beta_param = (2 - score_i) * 2, (1 + score_i) * 2
                CR = beta.rvs(alpha, beta_param)
                Ui = self.crossover(n_ind=self.D, ind=P[i], mutant=Vi, cross_proba=CR)
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
                if self.filter_init:
                    P[0], P[1], P[2] = r1, r2, indMax
                else:
                    P[0] = indMax
                scores = [fitness(train=self.train, test=self.test, columns=self.cols, ind=ind, target=self.target,
                          pipeline=self.pipeline, scoring=self.scoring, ratio=self.ratio, cv=self.cv)[0]for ind in P]
                bestScore, bestSubset, bestInd = add(scores=scores[:self.N], inds=np.asarray(P[:self.N]),
                                                     cols=self.cols)
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