
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

P_AVALUE = np.float64
P_AINDEX = np.intp

ctypedef np.intp_t AINDEX
ctypedef np.float64_t AVALUE


class TransitionMatrix:
    def __init__(self, matrix):
        self._matrix = matrix

    def enforce_symmetry(self):
        self._matrix += self._matrix.T

    def rownorm(self):
        rowsum = np.sum(self._matrix, axis=1)
        np.divide(
            self._matrix, rowsum[:, None],
            out=self._matrix
            )

    def nan_to_num(self):
        self._matrix = np.nan_to_num(self._matrix, copy=False)


class DiscreteTrajectory:

    def __init__(
            self, dtrajs, min_len_factor=10):
        self._process_input(dtrajs)
        self.reset()

        self._min_len_factor

    def __repr__(self):
        return f"{type(self)}()"

    @property
    def lag(self):
       return self._lag

    @lag.setter
    def lag(self, value):
        self.reset()
        self._lag = int(value)

    @property
    def min_len_factor(self):
       return self._min_len_factor

    @min_len_factor.setter
    def min_len_factor(self, value):
        self.reset()
        self._min_len_factor = int(value)

    def _process_input(self, dtrajs):
        self._dtrajs = [
            np.array(t, dtype=P_AINDEX, order="c")
            for t in dtrajs
        ]

    def reset(self):
        self._prepared_dtrajs = None
        self._n_states = None
        self._qminus = None
        self._qplus = None
        self._forward = None
        self._backward = None

    def estimate_transition_matrix(self):

        self.prepare_dtrajs()
        self.dtrajs_to_milestonings()
        self.milestonings_to_committers()

        transiton_matrix = np.zeros(
            (self._n_states, self._n_states), dtype=P_AVALUE
            )

        for index, forward in enumerate(self._forward):
            transiton_matrix += np.dot(
                forward[:len(forward) - self.lag].T,
                self._backward[index][self.lag:]
                )

        transiton_matrix = TransitionMatrix(transiton_matrix)
        transiton_matrix.enforce_symmetry()
        transiton_matrix.rownorm()
        transiton_matrix.nan_to_num()

        return transiton_matrix

    @staticmethod
    def trim_zeros(dtraj):
        nonzero = np.nonzero(dtraj)[0]
        try:
            first, last = nonzero[0], nonzero[-1]
            dtraj = dtraj[first:last + 1]
        except IndexError:
            dtraj = np.array([], dtype=P_AINDEX)

        return dtraj

    def prepare_dtrajs(self):
        if self._prepared_dtrajs is not None:
            return

        self._prepared_dtrajs = [
            self.trim_zeros(dtraj)
            for dtraj in self._dtrajs
        ]

        length = [len(x) for x in self._prepared_dtrajs]
        threshold = self._min_len_factor * self._lag

        self._n_states = max(np.max(x) for x in self._dtrajs)

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

    @staticmethod
    def _milestoning_to_committer(qminus, qplus, n_states):

        assert len(qminus) == len(qplus)

        forward = np.zeros((len(qminus), n_states), dtype=P_AINDEX)
        backward = np.zeros((len(qplus), n_states), dtype=P_AINDEX)

        for state, index in enumerate(range(n_states), 1):
            forward[qminus == state, index] = 1
            backward[qplus == state, index] = 1

        return forward, backward

    def dtrajs_to_milestonings(self):
        if (self._qminus is not None) or (self._qplus is not None):
            return

        self._qminus = []
        self._qplus = []

        for dtraj in self._dtrajs:
            qminus, qplus = self._dtraj_to_milestoning(dtraj)
            self._qminus.append(qminus)
            self._qplus.append(qplus)

    def milestonings_to_committers(self):
        if (self._forward is not  None) and (self._backward is not None):
            return

        self._forward = []
        self._backward = []

        for qminus, qplus in zip(self._qminus, self._qplus):
            forward, backward = self._milestoning_to_committer(
                qminus, qplus, self._n_states
                )
            self._forward.append(forward)
            self._backward.append(backward)

class CoresetMarkovStateModel:
    _DiscreteTrajectoryHandler = DiscreteTrajectory

    def __init__(
            self, dtrajs=None, lag=1, min_len_factor=10):
        if dtrajs is not None:
            dtrajs = self._DiscreteTrajectoryHandler(
                dtrajs, lag=lag, min_len_factor=min_len_factor
                )
        self.dtrajs = dtrajs

        self.transiton_matrix = None

    def __repr__(self):
        return f"{type(self).__name__}(dtrajs={self.dtrajs!s})"

    def estimate(self):
        self.transition_matrix = self.dtrajs.estimate_transition_matrix()
