import numpy as np
import pytest

from csmsm.estimator import CoresetMarkovStateModel
from csmsm.estimator import DiscreteTrajectory


class TestEstimator:

    def test_create_empty(self):
        model = CoresetMarkovStateModel()

        assert f"{model!r}" == "CoresetMarkovStateModel(dtrajs=None)"

    @pytest.mark.parametrize(
        "registered_key", [
            "test_case_1a",
            "test_case_1b",
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

    @pytest.mark.parametrize(
        "dtrajs,qminus,qplus",
        [
            (
                [np.array([1, 1, 2, 2, 0, 0, 0, 1, 1])],
                [np.array([1, 1, 2, 2, 2, 2, 2, 1, 1])],
                [np.array([1, 1, 2, 2, 1, 1, 1, 1, 1])],
            )
        ]
    )
    def test_milestoning(self, dtrajs, qminus, qplus):
        dtrajs = DiscreteTrajectory(dtrajs)
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
