
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

P_AVALUE = np.float64
P_AINDEX = np.intp

ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


class TransitionMatrix:
    def __init__(self, matrix):
        self.matrix = matrix


class DiscreteTrajectory:

    def __init__(self, dtrajs):
        self._dtrajs = dtrajs
        self._qminus = None
        self._qmax = None

    def __repr__(self):
        return f"{type(self)}()"

    def reset(self):
        self._qminus = None
        self._qplus = None

    def estimate_transition_matrix(self):
        return TransitionMatrix(None)

    @staticmethod
    def _dtraj_to_milestoning(dtraj):
        qminus = np.copy(dtraj)
        qplus = np.copy(dtraj)

        for index, state in enumerate(dtraj[1:], 1):
            if state == 0:
                qminus[index] = qminus[index - 1]

        for index, state in zip(
                range(len(dtraj) - 2, -1, -1),
                reversed(dtraj[:-1])):
            if state == 0:
                qplus[index] = qplus[index + 1]

        return qminus, qplus

    def dtrajs_to_milestoning(self):
        if (self._qminus is None) or (self._qplus is None):
            self._qminus = []
            self._qplus = []

            for dtraj in self._dtrajs:
                qminus, qplus = self._dtraj_to_milestoning(dtraj)
                self._qminus.append(qminus)
                self._qplus.append(qplus)


class CoresetMarkovStateModel:
    _DiscreteTrajectoryHandler = DiscreteTrajectory

    def __init__(self, dtrajs=None):
        if dtrajs is not None:
            dtrajs = self._DiscreteTrajectoryHandler(dtrajs)
        self.dtrajs = dtrajs

        self.transiton_matrix = None

    def __repr__(self):
        return f"{type(self).__name__}(dtrajs={self.dtrajs!s})"

    def estimate(self):
        self.transiton_matrix = self.dtrajs.estimate_transition_matrix
