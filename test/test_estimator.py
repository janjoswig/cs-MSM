from csmsm.estimator import CoresetMarkovStateModel


class TestEstimatior:

    def test_create(self):
        model = CoresetMarkovStateModel()

        assert f"{model!r}" == "CoresetMarkovStateModel(dtrajs=None)"
