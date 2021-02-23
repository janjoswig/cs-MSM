import numpy as np
import pytest

from csmsm.estimator import CoresetMarkovStateModel
from csmsm.estimator import TransitionMatrix
from csmsm.estimator import DiscreteTrajectory


class TestEstimator:

    def test_create_empty(self):
        model = CoresetMarkovStateModel()

        assert f"{model!r}" == "CoresetMarkovStateModel(dtrajs=None)"

    @pytest.mark.parametrize(
        "registered_key", [
            "test_case_1a",
            "test_case_1b",
            "test_case_1c",
            ]
        )
    def test_create_with_data_and_estimate(
            self, registered_key, registered_dtrajs, num_regression):
        model = CoresetMarkovStateModel(
            dtrajs=registered_dtrajs,
            min_len_factor=1
            )
        model.estimate()
        num_regression.check({
            "T": model.transition_matrix._matrix.flatten()
        })


class TestDiscreteTrajectory:

    def test_properties(self):
        dtrajs = DiscreteTrajectory([])
        assert isinstance(dtrajs.lag, int)
        assert isinstance(dtrajs.min_len_factor, int)
        dtrajs._prepared_dtrajs = []

        dtrajs.lag = 2
        assert dtrajs.lag == 2
        assert dtrajs._prepared_dtrajs is None

        dtrajs.min_len_factor = 5
        assert dtrajs.min_len_factor == 5

    @pytest.mark.parametrize(
        "dtrajs,attr_repr",
        [
            ([], "0 trajectories, lag=1, min_len_factor=10"),
            ([[1, 2]], "1 trajectory, lag=1, min_len_factor=10"),
            ([[1, 2], [1, 1]], "2 trajectories, lag=1, min_len_factor=10")
        ]
    )
    def test_repr(self, dtrajs, attr_repr):
        dtrajs = DiscreteTrajectory(dtrajs)
        assert f"{dtrajs!r}" == f"{type(dtrajs).__name__}({attr_repr})"

    @pytest.mark.parametrize(
        "traj,trimmed",
        [
            (
                np.array([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0]),
                np.array([1, 1, 2, 2, 0, 0, 0, 1, 1]),
            ),
            (
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([]),
            )
        ]
    )
    def test_trim_zeros(
            self, traj, trimmed):
        traj = DiscreteTrajectory.trim_zeros(traj)
        np.testing.assert_array_equal(
            traj,
            trimmed
            )

    @pytest.mark.parametrize(
        "dtrajs,prepared_dtrajs",
        [
            (
                [np.array([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0])],
                [np.array([1, 1, 2, 2, 0, 0, 0, 1, 1])],
            ),
            (
                [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])],
                [],
            )
        ]
    )
    def test_prepare_dtrajs(
            self, dtrajs, prepared_dtrajs):
        dtrajs = DiscreteTrajectory(dtrajs, min_len_factor=1)
        dtrajs.prepare_dtrajs()

        for index, prepared in enumerate(prepared_dtrajs):
            np.testing.assert_array_equal(
                dtrajs._prepared_dtrajs[index],
                prepared
            )

    def test_prepare_dtrajs_return_early(self):
        dtrajs = DiscreteTrajectory([])
        dtrajs._prepared_dtrajs = [[1]]
        dtrajs.prepare_dtrajs()

        assert dtrajs._prepared_dtrajs == [[1]]

    @pytest.mark.parametrize(
        "prepared_dtrajs,qminus,qplus",
        [
            (
                [np.array([1, 1, 2, 2, 0, 0, 0, 1, 1])],
                [np.array([1, 1, 2, 2, 2, 2, 2, 1, 1])],
                [np.array([1, 1, 2, 2, 1, 1, 1, 1, 1])],
            )
        ]
    )
    def test_milestoning(self, prepared_dtrajs, qminus, qplus):
        dtrajs = DiscreteTrajectory([])
        dtrajs._prepared_dtrajs = prepared_dtrajs
        dtrajs.dtrajs_to_milestonings()

        for index, milestoning in enumerate(qminus):
            np.testing.assert_array_equal(
                dtrajs._qminus[index], milestoning
            )
            np.testing.assert_array_equal(
                dtrajs._qplus[index], qplus[index]
            )

        dtrajs.reset()

        assert dtrajs._qminus is None
        assert dtrajs._qplus is None

    def test_milestoning_return_early(self):
        dtrajs = DiscreteTrajectory([])
        dtrajs._qminus = [[1]]
        dtrajs._qplus = [[2]]
        dtrajs.dtrajs_to_milestonings()

        assert dtrajs._qminus == [[1]]
        assert dtrajs._qplus == [[2]]

    @pytest.mark.parametrize(
        "qminus,qplus,forward,backward",
        [
            (
                [np.array([1, 1, 2, 2, 2, 2, 2, 1, 1])],
                [np.array([1, 1, 2, 2, 1, 1, 1, 1, 1])],
                [np.array([[1, 0],
                           [1, 0],
                           [0, 1],
                           [0, 1],
                           [0, 1],
                           [0, 1],
                           [0, 1],
                           [1, 0],
                           [1, 0]])],
                [np.array([[1, 0],
                           [1, 0],
                           [0, 1],
                           [0, 1],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0]])],
            )
        ]
    )
    def test_committer(self, qminus, qplus, forward, backward):
        dtrajs = DiscreteTrajectory([])
        dtrajs._qminus = qminus
        dtrajs._qplus = qplus
        dtrajs._states = [1, 2]
        dtrajs.milestonings_to_committers()

        for index, committer in enumerate(forward):
            np.testing.assert_array_equal(
                dtrajs._forward[index], committer
            )
            np.testing.assert_array_equal(
                dtrajs._backward[index], backward[index]
            )

    def test_committers_return_early(self):
        dtrajs = DiscreteTrajectory([])
        dtrajs._forward = [[1]]
        dtrajs._backward = [[2]]
        dtrajs.milestonings_to_committers()

        assert dtrajs._forward == [[1]]
        assert dtrajs._backward == [[2]]


class TestTransitionMatrix:

    @pytest.mark.parametrize(
        "matrix,expected",
        [
            (
                np.array([[0, 1],
                          [1, 0]]),
                [{0, 1}]
            ),
            (
                np.array([[1, 0],
                          [0, 1]]),
                [{0}, {1}]
            ),
            (
                np.array([[0.1, 0.9, 0.0],
                          [0.5, 0.2, 0.3],
                          [0.0, 0.3, 0.7]]),
                [{0, 1, 2}]
            ),
            (
                np.array([[0.1, 0.9, 0.0, 0.0, 0.0],
                          [0.8, 0.2, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.1, 0.6, 0.3],
                          [0.0, 0.0, 0.2, 0.5, 0.3],
                          [0.0, 0.0, 0.2, 0.4, 0.4]]),
                [{0, 1}, {2, 3, 4}]
            ),
            (
                np.array([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                          [0.8, 0.2, 0.0, 0.2, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.1, 0.0, 0.1, 0.5, 0.3],
                          [0.0, 0.0, 0.0, 0.2, 0.5, 0.3],
                          [0.0, 0.0, 0.0, 0.2, 0.4, 0.4]]),
                [{0, 1, 3, 4, 5}]
            )
        ]
    )
    def test_connected_sets(self, matrix, expected):
        transition_matrix = TransitionMatrix(matrix)
        returned = transition_matrix.connected_sets()
        assert returned == expected

    def test_eig(self):
        transition_matrix = TransitionMatrix(np.diag((1, 2, 3)))

        np.testing.assert_array_equal(
            transition_matrix.eigenvalues,
            np.array([3, 2, 1])
            )

        np.testing.assert_array_equal(
            transition_matrix.eigenvectors_right,
            np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
            )

        np.testing.assert_array_equal(
            transition_matrix.eigenvectors_left,
            np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
            )
