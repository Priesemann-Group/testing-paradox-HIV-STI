import pytest
import numpy as np
import jax.numpy as jnp
from testing_artefacts_pharmaco_multipath.STI.model_STI import (
    m_logistic,
    m_exponential,
    m,
    lambda_STI,
    infect_ia,
    infect_is,
    model,
    setup_model,
)


@pytest.fixture
def args():
    return {
        "m_max": 1.0,
        "H_thres": 0.5,
        "m_eps": 0.01,
        "H": 0.3,
        "min_exp": 0.1,
        "max_exp": 1.0,
        "tau_exp": 0.5,
        "m_function": "exponential",
        "lambda_0_a": 0.1,
        "c": 0.2,
        "beta_HIV": 0.3,
        "P_HIV": 0.1,
        "lambda_P": 0.4,
        "asymptomatic": 0.5,
        "beta_STI": 0.6,
        "gamma_STI": 0.7,
        "lambda_0": 0.8,
        "gammaT_STI": 0.9,
        "mu": 0.01,
    }


@pytest.fixture
def y():
    return {"Ia_STI": 10, "Is_STI": 5, "S_STI": 100, "T_STI": 2}


def test_m_logistic(args):
    result = m_logistic(args)
    assert isinstance(result, jnp.ndarray)


def test_m_exponential(args):
    result = m_exponential(args)
    assert isinstance(result, jnp.ndarray)


def test_m(args):
    args["m_function"] = "logistic"
    result = m(args)
    assert isinstance(result, jnp.ndarray)

    args["m_function"] = "exponential"
    result = m(args)
    assert isinstance(result, jnp.ndarray)


def test_lambda_STI(args):
    result = lambda_STI(args)
    assert isinstance(result, jnp.ndarray)


def test_infect_ia(y, args):
    result = infect_ia(y, args)
    assert isinstance(result, jnp.ndarray)


def test_infect_is(y, args):
    result = infect_is(y, args)
    assert isinstance(result, jnp.ndarray)


def test_model(args):
    t = 0
    y = [100, 10, 5, 2]
    result = model(t, y, args)
    assert isinstance(result, list)


def test_setup_model(args):
    y0 = [100, 10, 5, 2]
    integrator = setup_model(args, y0)
    assert integrator is not None
