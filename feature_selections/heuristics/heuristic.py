import abc
import os
import psutil
import joblib
from sklearn.base import ClassifierMixin, RegressorMixin

from feature_selections import FeatureSelection
from utility import fitness


class Heuristic(FeatureSelection):
    """
    Parent class for all heuristics

    Args:
        N (int): The number of individuals evaluated per iteration
    """
    def __init__(self, name, target, pipeline, train, test=None, cv=None, drops=None, scoring=None, N=None, Gmax=None,
                 Tmax=None, ratio=None, suffix=None, verbose=None, output=None):
        super().__init__(name, target, pipeline, train, test, cv, drops, scoring, Gmax, Tmax, ratio, suffix, verbose,
                         output)
        self.N = N if N is not None else 100

    @abc.abstractmethod
    def start(self, pid):
        pass

    @staticmethod
    def pprint_(print_out, name, pid, maxi, best, mean, feats, time_exe, time_total, entropy, g, cpt, verbose):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}" \
                  "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}    entropy : {:2.3f}" \
            .format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt, entropy)
        print_out = print_out + display
        if verbose:
            print(display)
        return print_out

    @staticmethod
    def sprint_(print_out, name, pid, maxi, best, mean, feats, time_exe, time_total, g, cpt, verbose):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4f}    features: {:6d}    best: {:2.4f}" \
                  "   mean: {:2.4f}    G time: {}    T time: {}    last: {:6d}" \
            .format(name, pid, g, maxi, feats, best, mean, time_exe, time_total, cpt)
        print_out = print_out + display
        if verbose:
            print(display)
        return print_out

    def save(self, name, bestInd, g, t, last, specifics, out):
        a = os.path.join(os.path.join(self.path, 'results.txt'))
        with open(a, "w") as f:
            try:
                method = self.pipeline.steps[-1][1].__class__.__name__
            except Exception:
                method = str(self.pipeline)
            bestSubset = [self.cols[i] for i in range(len(self.cols)) if bestInd[i]]
            score_train, y_true, y_pred = fitness(
                train=self.train,
                test=self.test,
                columns=self.cols,
                ind=bestInd,
                target=self.target,
                pipeline=self.pipeline,
                scoring=self.scoring,
                ratio=0,
                cv=self.cv
            )
            if isinstance(self.pipeline.steps[-1][1], ClassifierMixin):
                tp, tn, fp, fn = self.calculate_confusion_matrix_components(y_true, y_pred)
                string_tmp = f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}" + os.linesep
            elif isinstance(self.pipeline.steps[-1][1], RegressorMixin):
                string_tmp = f"Regression residuals (first 5): {(y_true - y_pred).head().tolist()}" + os.linesep
            string = (
                    f"Heuristic: {name}" + os.linesep +
                    f"Population: {self.N}" + os.linesep +
                    f"Generations: {self.Gmax}" + os.linesep +
                    f"Generations Performed: {g}" + os.linesep +
                    f"Latest Improvement: {last}" + os.linesep +
                    f"Latest Improvement (Ratio): {1 - (last / g)}" + os.linesep +
                    f"Cross-validation strategy: {str(self.cv)}" + os.linesep +
                    specifics +
                    f"Method: {method}" + os.linesep +
                    f"Best Score: {score_train}" + os.linesep +
                    string_tmp +
                    f"Best Subset: {bestSubset}" + os.linesep +
                    f"Number of Features: {len(bestSubset)}" + os.linesep +
                    f"Number of Features (Ratio): {len(bestSubset) / len(self.cols)}" + os.linesep +
                    f"Execution Time: {round(t.total_seconds())} ({t})" + os.linesep +
                    f"Memory: {psutil.virtual_memory()}"
            )
            f.write(string)
        a = os.path.join(self.path, 'log.txt')
        with open(a, "a") as f:
            f.write(out)
        # Pipeline saving
        pipeline_path = os.path.join(self.path, 'pipeline.joblib')
        joblib.dump(self.pipeline, pipeline_path)
