
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free


P_AVALUE = np.float64
P_AINDEX = np.intp


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

    def largest_connected_set(self):
        rowsum = np.sum(self._matrix, axis=1)
        nonzerorows = np.nonzero(rowsum)[0]

        original_set = set(range(len(self._matrix)))
        connected_sets = []
        while original_set:
            root = next(x for x in original_set if x in nonzerorows)
            current_set = set([root])
            added_set = set([root])
            original_set.remove(root)
            while added_set:
                added_set_support = set()
                for cluster in added_set:
                    nonzero_transitions = np.nonzero(self._matrix[cluster])[0]
                    for connected_cluster in nonzero_transitions:
                        if connected_cluster not in current_set:
                            current_set.add(connected_cluster)
                            added_set_support.add(connected_cluster)
                            original_set.remove(connected_cluster)
                added_set = added_set_support.copy()
            connected_sets.append(current_set)

        return connected_sets


class DiscreteTrajectory:

    _TransitionMatrixHandler = TransitionMatrix

    def __init__(
            self, dtrajs, lag=1, min_len_factor=10):
        self._process_input(dtrajs)

        self._lag = int(lag)
        self._min_len_factor = int(min_len_factor)

        self.reset()

    def __repr__(self):
        length = len(self._dtrajs)
        if length == 1:
            length_str = "1 trajectory"
        else:
            length_str = f"{length} trajectories"

        attr_repr = ", ".join(
            [
                length_str,
                f"lag={self.lag}",
                f"min_len_factor={self.min_len_factor}"
                ]
        )
        return f"{type(self).__name__}({attr_repr})"

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

    @staticmethod
    def _committers_to_transition_matrix(forward, backward, n_states, lag):
        transition_matrix = np.zeros(
            (n_states, n_states), dtype=P_AVALUE
            )

        for index, f in enumerate(forward):
            transition_matrix += np.dot(
                f[:len(f) - lag].T,
                backward[index][lag:]
                )

        return transition_matrix


    def estimate_transition_matrix(self):

        self.prepare_dtrajs()
        self.dtrajs_to_milestonings()
        self.milestonings_to_committers()

        mass_matrix = self._committers_to_transition_matrix(
            forward=self._forward,
            backward=self._backward,
            n_states=self._n_states,
            lag=0
        )
        transition_matrix = self._committers_to_transition_matrix(
            forward=self._forward,
            backward=self._backward,
            n_states=self._n_states,
            lag=self.lag
        )

        transition_matrix = self._TransitionMatrixHandler(transition_matrix)
        transition_matrix.enforce_symmetry()
        transition_matrix.rownorm()
        transition_matrix.nan_to_num()

        mass_matrix = self._TransitionMatrixHandler(mass_matrix)
        mass_matrix.enforce_symmetry()
        mass_matrix.rownorm()
        mass_matrix.nan_to_num()

        connected_sets = transition_matrix.largest_connected_set()
        set_size = [len(x) for x in connected_sets]
        largest_connected = list(connected_sets[np.argmax(set_size)])

        transition_matrix._matrix = transition_matrix._matrix[
            tuple(np.meshgrid(largest_connected, largest_connected))
            ].T
        mass_matrix._matrix = mass_matrix._matrix[
            tuple(np.meshgrid(largest_connected, largest_connected))
            ].T

        # Weight T with the inverse M
        transition_matrix._matrix = np.dot(
            transition_matrix._matrix,
            np.linalg.inv(mass_matrix._matrix))

        return transition_matrix

    @staticmethod
    def trim_zeros(dtraj):
        nonzero = np.nonzero(dtraj)[0]

        length = nonzero.shape[0]
        if length == 0:
           return np.array([], dtype=P_AINDEX)

        first, last = nonzero[0], nonzero[length - 1]

        return dtraj[first:last + 1]

    def prepare_dtrajs(self):
        if self._prepared_dtrajs is not None:
            return

        self._prepared_dtrajs = [
            self.trim_zeros(dtraj)
            for dtraj in self._dtrajs
        ]

        length = [len(dtraj) for dtraj in self._prepared_dtrajs]
        threshold = self._min_len_factor * self._lag

        empty = []
        tooshort = []
        for index, l in enumerate(length):
            if l == 0:
                empty.append(index)
            elif l < threshold:
                tooshort.append(index)

        self._prepared_dtrajs = [
            dtraj for index, dtraj
            in enumerate(self._prepared_dtrajs)
            if length[index] >= threshold
            ]

        highest_states = [np.max(x) for x in self._prepared_dtrajs] + [0]
        self._n_states = max(highest_states)

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
        if (self._qminus is not None) and (self._qplus is not None):
            return

        self._qminus = []
        self._qplus = []

        for dtraj in self._prepared_dtrajs:
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
