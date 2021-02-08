
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

P_AVALUE = np.float64
P_AINDEX = np.intp

ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


class DiscreteTrajectory:

    def __init__(self, dtrajs):
        self.dtrajs = dtrajs

    def __repr__(self):
        return f"{type(self)}()"

class CoresetMarkovStateModel:
    _DiscreteTrajectoryHandler = DiscreteTrajectory

    def __init__(self, dtrajs=None):
        if dtrajs is not None:
            dtrajs = self._DiscreteTrajectoryHandler(dtrajs)
        self.dtrajs = dtrajs

    def __repr__(self):
        return f"{type(self).__name__}(dtrajs={self.dtrajs!s})"
