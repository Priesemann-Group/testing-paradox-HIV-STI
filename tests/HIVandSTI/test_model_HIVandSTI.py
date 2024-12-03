if False: 
    import pytest
    import numpy as np
    import jax.numpy as jnp
    from tapm.HIVandSTI.model_HIVandSTI import (
        m_logistic,
        m_exponential,
        m,
        lambda_a,
        beta_STI,
        beta_HIV,
        phi_H_eff,
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
            "lambda_0": 0.1,
            "c": 0.2,
            "beta_HIV": 0.3,
            "P_HIV": 0.1,
            "lambda_P": 0.4,
            "beta_STI": 0.5,
            "beta_HIV": 0.6,
            "rho": 0.7,
            "lambda_s": 0.8,
            "nu": 0.9,
            "alpha": 0.1,
            "mu": 0.01,
            "psi": 0.5,
            "gamma_STI": 0.7,
            "gammaT_STI": 0.9,
            "tau": 0.1,
            "phi_r": 0.2,
            "phi_min": 1.0,
            "phi_max": 1.0,
            "lambda_ARV": 0.3,
        }


    @pytest.fixture
    def y():
        return {
            "Ia_STI": 10,
            "Is_STI": 5,
            "S_STI": 100,
            "T_STI": 2,
            "S_HIV": 90,
            "E_HIV": 3,
            "I_HIV": 7,
            "T_HIV": 1,
            "P_HIV": 4,
            "H": 0.2,
            "h": 0.1,
        }


    def test_m_logistic(y, args):
        result = m_logistic(y, args)
        assert isinstance(result, jnp.ndarray)


    def test_m_exponential(y, args):
        result = m_exponential(y, args)
        assert isinstance(result, jnp.ndarray)


    def test_m(y, args):
        args["m_function"] = "logistic"
        result = m(y, args)
        assert isinstance(result, jnp.ndarray)

        args["m_function"] = "exponential"
        result = m(y, args)
        assert isinstance(result, jnp.ndarray)


    def test_lambda_a(y, args):
        result = lambda_a(y, args)
        assert isinstance(result, jnp.ndarray)


    def test_beta_STI(y, args):
        result = beta_STI(y, args)
        assert isinstance(result, jnp.ndarray)


    def test_beta_HIV(y, args):
        result = beta_HIV(y, args)
        assert isinstance(result, jnp.ndarray)


    def test_phi_H_eff(y, args):
        result = phi_H_eff(y, args)
        result = jnp.array(result)
        assert isinstance(result, jnp.ndarray)


    def test_model(args):
        t = 5
        y = {
            "S_STI": 100,
            "Ia_STI": 10,
            "Is_STI": 5,
            "T_STI": 2,
            "S_HIV": 90,
            "E_HIV": 3,
            "I_HIV": 7,
            "T_HIV": 1,
            "P_HIV": 4,
            "h": 0.2,
            "H": 0.1,
        }
        func_t = lambda x: x
        result = model(t, y, (func_t ,args))
        assert isinstance(result, dict)


    def test_setup_model(args):
        y0 = [100, 10, 5, 2, 90, 3, 7, 1, 4, 0.2, 0.1, 0.3]
        integrator = setup_model(args, y0)
        assert integrator is not None
