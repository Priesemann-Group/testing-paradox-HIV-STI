import pytest
import numpy as np
import jax.numpy as jnp
import diffrax
from icomo import diffeqsolve
from tapm.HIVandSTI.model_HIVandSTI import (
    main_model,
    m,
    foi_HIV,
    foi_STI,
    contact_matrix,
    hazard,
    prep_fraction,
    lambda_a,
    lambda_s
)

@pytest.fixture
def args():
    return {
        "beta_HIV": 0.6341 / 365.0,
        "N_0": np.array([0.451, 0.353, 0.125, 0.071]),
        "mu": 1 / 45 / 365.0,
        "Omega": 1 - 0.86,
        "c": np.array([0.13, 1.43, 5.44, 18.21]) / 365.0,
        "h": np.array([0.62, 0.12, 0.642, 0.0]),
        "epsilon": 0.01,
        "epsilonP": 0.62 / 2,
        "Lambda": np.array([0.25] * 4),
        "omega": 0.5,
        "phis": np.array([0.05, 0.05, 0.05, 0.05]),
        "taus": np.array([0.3, 0.3, 0.3, 0.3]),
        "tau_p": 0.95,
        "rhos": np.array([0.142, 8.439, 1.184, 1.316]) / 365.0,
        "gammas": np.array([8.21, 54.0, 2.463, 2.737]) / 365.0,
        "k_on": 0.3,
        "k_off": 5.0,
        "asymptomatic": 0.85,
        "beta_STI": 0.0016 * 7.0,
        "lambda_0": 1 / 14.0,
        "lambda_P": 2 / 365,
        "m_function": "exponential",
        "min_exp": 0.0,
        "max_exp": 1.0,
        "tau_exp": 0.2,
        "m_eps": 0.01,
        "Sigma": 0.01 / 365,
        "scaling_factor_m_eps": 1.0,
        "gamma_STI": 1 / 1.32 / 365.0,
        "gammaT_STI": 1 / 7.0,
        "contact": 50.0,
        "delay": np.array([20.0]),
    }


@pytest.fixture
def y0():
    return {
        "S": np.array([0.49874, 0.4919, 0.42845, 0.42395]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "SP": np.array([0.49874, 0.4919, 0.42845, 0.42395]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "I1": np.array([0.00001, 0.0001, 0.001, 0.001]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "IP": np.array([0.00001, 0.0001, 0.0001, 0.0001]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "I2": np.array([0.001, 0.001, 0.01, 0.01]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "I3": np.array([0.0001, 0.001, 0.01, 0.01]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "I4": np.array([0.0001, 0.001, 0.001, 0.01]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "A1": np.array([0.0001, 0.001, 0.001, 0.01]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "A2": np.array([0.001, 0.01, 0.1, 0.1]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "A3": np.array([0.0001, 0.001, 0.01, 0.01]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "A4": np.array([0.0001, 0.001, 0.01, 0.001]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "D": np.array([0.0, 0.0, 0.0, 0.0]) * np.array([0.451, 0.353, 0.125, 0.071]),
        "S_STI": 0.65 * np.array([0.451, 0.353, 0.125, 0.071]),
        "Ia_STI": 0.15 * np.array([0.451, 0.353, 0.125, 0.071]),
        "Is_STI": 0.15 * np.array([0.451, 0.353, 0.125, 0.071]),
        "T_STI": 0.05 * np.array([0.451, 0.353, 0.125, 0.071]),
        "H": np.array([0.0, 0.0, 0.0, 0.0]) * np.array([0.451, 0.353, 0.125, 0.071]),
    }


def test_m(args, y0):
    result = m(args, y0)
    assert isinstance(result, jnp.ndarray)


def test_foi_HIV(y0, args):
    result = foi_HIV(y0, args)
    assert isinstance(result, jnp.ndarray)


def test_foi_STI(y0, args):
    result = foi_STI(y0, args)
    assert isinstance(result, jnp.ndarray)


def test_contact_matrix(args):
    result = contact_matrix(args)
    assert isinstance(result, jnp.ndarray)


def test_hazard(y0, args):
    result = hazard(y0, args)
    assert isinstance(result, jnp.ndarray)


def test_prep_fraction(y0):
    result = prep_fraction(y0)
    assert isinstance(result, jnp.ndarray)


def test_lambda_a(y0, args):
    result = lambda_a(y0, args)
    assert isinstance(result, jnp.ndarray)


def test_lambda_s(y0, args):
    result = lambda_s(y0, args)
    assert isinstance(result, jnp.ndarray)


def test_main_model(y0, args):
    output = main_model(0, y0, args)
    assert isinstance(output, dict)


def test_diffeqsolve(args, y0):
    output = diffeqsolve(
        args=args,
        ODE=main_model,
        y0=y0,
        ts_out=np.linspace(0, 365 * 20, 365 * 20 + 1),
        max_steps=365 * 20 + 1,
    )
    assert isinstance(output, diffrax.Solution)