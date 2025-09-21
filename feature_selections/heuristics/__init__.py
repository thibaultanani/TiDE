"""Convenience exports for the TiDE heuristic feature-selection strategies.

Each heuristic exposes the same ``FeatureSelection`` interface but uses
different search dynamics.  Typical use cases:

* ``Genetic`` / ``Differential`` / ``Nmbde`` / ``Pbil`` / ``Tide`` – population
  based metaheuristics; tune ``N`` (population size) and ``Gmax`` (generations)
  alongside the strategy-specific hyper-parameters (``mutation``, ``entropy``,
  ``gamma``, etc.).
* ``ForwardSelection`` / ``BackwardSelection`` / ``LocalSearch`` /
  ``SimulatedAnnealing`` / ``Tabu`` – single-solution heuristics; focus on
  per-step budgets like ``Tmax`` and neighbourhood controls (``nb`` or
  temperature settings).
* ``Random`` – baseline random sampling useful for smoke tests or as an upper
  bound on stochastic noise.

All classes return the same bookkeeping tuple as documented on
``Heuristic.start``.
"""

from .population_based import Genetic
from .population_based import Differential
from .population_based import Pbil
from .population_based import Tide
from .population_based import Nmbde
from .single_solution import ForwardSelection
from .single_solution import BackwardSelection
from .single_solution import LocalSearch
from .single_solution import SimulatedAnnealing
from .single_solution import Tabu
from .other import Random
