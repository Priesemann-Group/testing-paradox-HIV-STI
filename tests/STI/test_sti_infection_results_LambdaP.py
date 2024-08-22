import pytest
import numpy as np
from unittest.mock import MagicMock
from tapm.STI.sti_infection_results_LambdaP import (
    compute_sti_infections,
)


@pytest.fixture
def setup_data():
    Hs = [0.1, 0.2]
    Ps = [0.1, 0.2]
    lambda_P_values = [0.1, 0.2]
    yin_constant = {"Ia_STI": 10, "Is_STI": 5, "S_STI": 100, "T_STI": 2}
    yin_variable = False
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
    return (
        Hs,
        Ps,
        lambda_P_values,
        yin_constant,
        yin_variable,
        args,
        integrator,
        model_STI,
    )


def test_compute_sti_infections(setup_data):
    Hs, Ps, lambda_P_values, yin_constant, yin_variable, args, integrator, model_STI = (
        setup_data
    )
    results = compute_sti_infections(
        Hs, Ps, lambda_P_values, yin_constant, yin_variable, args, integrator, model_STI
    )
    assert isinstance(results, dict)
    for lambda_P in lambda_P_values:
        assert lambda_P in results
        assert "res_Ia" in results[lambda_P]
        assert "res_Is" in results[lambda_P]
        assert "res_T" in results[lambda_P]
        assert "res_infections" in results[lambda_P]
        assert "res_asymp_infections" in results[lambda_P]
        assert "res_symp_infections" in results[lambda_P]
        assert "res_tests" in results[lambda_P]
        assert "res_asymp_tests" in results[lambda_P]
        assert "res_symp_tests" in results[lambda_P]
        assert "check" in results[lambda_P]
