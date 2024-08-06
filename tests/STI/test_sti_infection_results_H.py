import pytest
import numpy as np
from unittest.mock import MagicMock
from tapm.STI.sti_infection_results_H import (
    compute_sti_infections,
)


@pytest.fixture
def setup_data():
    Hs = [0.1, 0.2]
    Ps = [0.1, 0.2]
    lambda_P_values = [0.1, 0.2]
    y0 = {"S_STI": 100, "Ia_STI": 10, "Is_STI": 5, "T_STI": 2}
    args = {"lambda_s": 0.1, "H": 0.1, "P_HIV": 0.1, "lambda_P": 0.1}
    integrator = MagicMock()
    integrator.return_value = {
        "Ia_STI": np.array([10, 9, 8]),
        "Is_STI": np.array([5, 4, 3]),
        "T_STI": np.array([2, 1, 0]),
        "S_STI": np.array([100, 99, 98]),
    }
    model_STI = MagicMock()
    model_STI.infect_is.return_value = 0.1
    model_STI.infect_ia.return_value = 0.1
    model_STI.lambda_STI.return_value = 0.1
    return Hs, Ps, lambda_P_values, y0, args, integrator, model_STI


def test_compute_sti_infections(setup_data):
    Hs, Ps, lambda_P_values, y0, args, integrator, model_STI = setup_data
    results = compute_sti_infections(
        Hs, Ps, lambda_P_values, y0, args, integrator, model_STI
    )
    assert isinstance(results, dict)
    for H in Hs:
        assert H in results
        assert "res_Ia" in results[H]
        assert "res_Is" in results[H]
        assert "res_T" in results[H]
        assert "res_infections" in results[H]
        assert "res_asymp_infections" in results[H]
        assert "res_symp_infections" in results[H]
        assert "res_tests" in results[H]
        assert "res_asymp_tests" in results[H]
        assert "res_symp_tests" in results[H]
        assert "check" in results[H]
