import numpy as np
import pytest

from csmsm.estimator import CoresetMarkovStateModel
from csmsm.estimator import DiscreteTrajectory


class TestEstimator:

    def test_create_empty(self):
        model = CoresetMarkovStateModel()

        assert f"{model!r}" == "CoresetMarkovStateModel(dtrajs=None)"

    @pytest.mark.parametrize(
        "data",
        [
            [[np.array([0, 1, 2, 1, 1, 0, 0, 2, 2, 2, 0])]]
        ]
        )
    def test_create_with_data_and_estimate(self, data):
        model = CoresetMarkovStateModel(dtrajs=data)
        model.estimate()


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
        dtrajs.dtrajs_to_milestoning()

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
