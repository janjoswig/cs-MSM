import numpy as np
import pytest

from csmsm.estimator import CoresetMarkovStateModel
from csmsm.estimator import DiscreteTrajectory
from csmsm.estimator import P_AVALUE, P_AINDEX


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
        model = CoresetMarkovStateModel(dtrajs=registered_dtrajs)
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
                [np.array([])],
            )
        ]
    )
    def test_prepare_dtrajs(
            self, dtrajs, prepared_dtrajs):
        dtrajs = DiscreteTrajectory(dtrajs)
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
        dtrajs._n_states = 2
        dtrajs.milestonings_to_committers()

        for index, committer in enumerate(forward):
            np.testing.assert_array_equal(
                dtrajs._forward[index], committer
            )
            np.testing.assert_array_equal(
                dtrajs._backward[index], backward[index]
            )
