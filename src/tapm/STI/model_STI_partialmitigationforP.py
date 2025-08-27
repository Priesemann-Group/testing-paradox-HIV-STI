# Parameters from GannaRozhnova's paper (Elimination prospects of the Dutch HIV epidemic among men who have sex with men in the era of
#  pre-exposure prophylaxis)  https://pubmed.ncbi.nlm.nih.gov/30379687/


import numpy as np 
import icomo
import jax.numpy as jnp
import logging
from jax import debug

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Global flag to track logging
logged_exp_logis = False






def m(args, y):
    """
    Exponential function with three parameters: minimum value, maximum value, and rate/tau.

    Args:
    args (dict): A dictionary containing the parameters 'H', 'min_exp', 'max_exp', and 'tau_exp'.
    H (float): The current value of 'H'.

    Returns:
    float: The output of the exponential function.
    """
    global logged_exp_logis
    logger.debug("Calculating self-regulation factor factor 'm' using exponential function")

    if not logged_exp_logis:
        logger.info("Using exponential function to calculate m")
        logger.info(
            "Parameters: min_exp = %s, max_exp = %s",
            args["min_exp"],
            args["max_exp"],
        )
        logged_exp_logis = True
    res = args["min_exp"] + (args["max_exp"] - args["min_exp"]) * (1 - jnp.exp(-hazard(y, args) / args["H_thres"]))
    return res



def foi_STI(y, args):
    """
    Calculates the force of infection for STIs.

    Args:
        y (dict): A dictionary containing the state variables of the model. 
        args (Any): Additional parameters required for the calculation, 

    Returns:
        ndarray: A 1x4 array representing the force of infection for each group.
    """

    I_eff = y["Ia_STI"] + y["Is_STI"]
    foi = args["beta_STI"] * ((1 - m(args, y))*(1 - prep_fraction(y, args)) + (1 - args["xi"]*m(args, y))*prep_fraction(y, args)) * I_eff
    return foi


def hazard(y, args):
    """
    Calculates the hazard from the HIV model used for risk-perception.
    Right now the hazard is just the fraction of people on ART in the HIV model.

    Returns:
        Value as float64.
    """
    return args["H"] # Risk perception H will be somehow homogeneous across the population

def prep_fraction(y, args):
    """
    Calculates fraction of people on PrEP in the HIV model.

    Returns:
        Value as float64.
    """
    return args["P"]


def lambda_a(y, args):
    """
    Calculates STI testing rate for asymtomatic infected.

    Returns:
        Array of rates with dimension [1, risk groups].
    """
    risk_induced_testing = args["c"] * args["beta_HIV"] * (1 - m(args, y)) * hazard(y, args) * (1 - prep_fraction(y, args))
    prep_induced_testing = args["lambda_P"] * prep_fraction(y, args)
    return risk_induced_testing + prep_induced_testing


def lambda_s(y, args):
    """
    Calculates STI testing rate for symtomatic infected.

    Returns:
        Array of rates with dimension [1, risk groups].
    """
    return args["lambda_s"] + lambda_a(y, args)



def main_model(t, y, args):
    cm = icomo.CompModel(y)  # Initialize the compartmental model


    # STI dynamics-------------------------------------------------------------------------------------------------------------------------------------------------
    cm.flow("S_STI",  "Ia_STI", (args["psi"])   * foi_STI(y, args))     # Susceptible to asymptomatic
    cm.flow("S_STI",  "Is_STI", (1-args["psi"]) * foi_STI(y, args))     # Susceptible to symptomatic
    cm.flow("Ia_STI", "S_STI",  args["gamma_STI"])                      # Asymptomatic to susceptible (recovery)
    cm.flow("Ia_STI", "T_STI",  lambda_a(y,args))                       # Asymptomatic to tested and treatment
    cm.flow("Is_STI", "T_STI",  lambda_s(y,args))                       # Symptomatic to tested and treatment
    cm.flow("T_STI",  "S_STI",  args["gammaT_STI"])                     # Treatment to susceptible (immunity loss)
    cm.dy["S_STI"] -= args["Sigma"]                                     # Influx
    cm.dy["Ia_STI"] += args["psi"] * args["Sigma"]                      # Influx
    cm.dy["Is_STI"] += (1-args["psi"]) * args["Sigma"]                  # Influx
    # Vital dynamics
    cm.flow("Ia_STI", "S_STI", args["mu"])  # Death/removal from asymptomatic
    cm.flow("Is_STI", "S_STI", args["mu"])  # Death/removal from symptomatic
    cm.flow("T_STI", "S_STI", args["mu"])  # Death/removal from treatment





    return cm.dy

