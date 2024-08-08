# This python file contains the model definition for the coupled HIV and STI dynamics in the model.
# The model is defined as a set of differential equations that describe the flow of individuals between different compartments.

# compartments stored in y:
# S_HIV: Susceptible to HIV
# E_HIV: Exposed to HIV
# I_HIV: Infected with HIV
# T_HIV: Tested and treated for HIV
# P: Protected (through PrEP)
# S_STI: Susceptible to STI
# Ia_STI: Asymptomatic STI cases
# Is_STI: Symptomatic STI cases
# T_STI: Tested and treated for STI
# H: hazard
# h: help variable for hazard
# phi_H: [TODO: name?]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import jax
import icomo
import logging
import jax.numpy as jnp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to track logging
logged_exp_logis = False


# Function to calculate the modulating factor 'm' based on provided arguments
def m_logistic(args):
    """
    Calculate the self-regulation factor 'm' based on the given arguments.

    Args:
        args (dict): A dictionary containing the following keys:
            - "m_max" (float): The maximum value of 'm'.
            - "H_thres" (float): The threshold value of 'H'.
            - "m_eps" (float): A small positive constant.
            - "H" (float): The current value of 'H'.

    Returns:
        float: The calculated value of 'm'.

    """
    global logged_exp_logis
    logger.debug("Calculating modulating factor 'm'")
    if not logged_exp_logis:
        logger.info("Using logistic function to calculate m")
        logger.info(
            "Parameters: m_max = %s, H_thres = %s, m_eps = %s, H = %s",
            args["m_max"],
            args["H_thres"],
            args["m_eps"],
        )
        logged_exp_logis = True
    return args["m_max"] - args["m_max"] / args["H_thres"] * args[
        "m_eps"
    ] * jax.numpy.log(1 + jax.numpy.exp((args["H_thres"] - args["H"]) / args["m_eps"]))


def m_exponential(args):
    """
    Exponential function with three parameters: minimum value, maximum value, and rate/tau.

    Args:
    args (dict): A dictionary containing the parameters 'H', 'min_exp', 'max_exp', and 'tau_exp'.

    Returns:
    float: The output of the exponential function.
    """
    global logged_exp_logis
    logger.debug(
        "Calculating self-regulation factor factor 'm' using exponential function"
    )
    H = args["H"]
    min_exp = args["min_exp"]
    max_exp = args["max_exp"]
    tau_exp = args["tau_exp"]

    if not logged_exp_logis:
        logger.info("Using exponential function to calculate m")
        logger.info(
            "Parameters: min_exp = %s, max_exp = %s, tau_exp = %s",
            min_exp,
            max_exp,
            tau_exp,
        )
        logged_exp_logis = True

    return min_exp + (max_exp - min_exp) * (1 - jnp.exp(-H / tau_exp))


def m(args):
    """
    Select the appropriate function to calculate the self-regulation factor 'm' based on the arguments.

    Args:
        args (dict): A dictionary containing the arguments for the calculation.

    Returns:
        The result of the calculation.
    """
    if args["m_function"] == "exponential":
        return m_exponential(args)
    elif args["m_function"] == "logistic":
        return m_logistic(args)
    else:
        raise ValueError("Invalid m_function specified in args")


# Function to calculate the testing rate of STI
def lambda_a(y, args):
    logger.debug("Calculating testing rate of STI")
    return (
        args["lambda_0"]  # Baseline test rate
        + args["c"]  # contacts
        * (1 - m(args))
        * args["beta_HIV"]
        * args["H"]
        * (1 - args["P_HIV"])  # HIV dependent term
        + args["lambda_P"]
        * args["P_HIV"]  # Proportional infection rate due to HIV prevalence
    )


# Function to calculate STI infection rate
def beta_STI(y, args):
    logger.debug("Calculating beta_STI")
    return args["beta_0_STI"] * ((1 - m(args)) * (1 - y["P"]) + y["P"])


# Function to calculate HIV infection rate
def beta_HIV(args):
    logger.debug("Calculating beta_HIV")
    return args["beta_0_HIV"] * (1 - m(args))


# Function to calculate effective [TODO: name?]
def phi_H_eff(y, args):
    logger.debug("Calculating phi_H_eff")
    return y["phi_H"] * (y["S_STI"] + y["P"])  # TODO: S_STI or S_HIV?


# Main model function that defines the differential equations of the system
def model(t, y, args):
    logger.debug("Defining the differential equations of the system HIVandSTI")
    cm = icomo.CompModel(y)  # Initialize the compartmental model

    # Basic HIV dynamics
    cm.flow("S_HIV", "E_HIV", beta_HIV(args) * y["I_HIV"])  # Susceptible to exposed
    cm.flow("E_HIV", "I_HIV", args["rho"])  # Exposed to infected
    cm.flow("I_HIV", "T_HIV", args["lambda_s"])  # Infected to tested and treatment
    cm.flow(
        "T_HIV", "I_HIV", args["nu"]
    )  # Testes and treated to infected, dropout/ need of new testing
    cm.flow(
        "P", "S_HIV", args["alpha"] * (phi_H_eff(y, args) / y["P"] + 1)
    )  # Protected to tested and treatment

    # Vital dynamics HIV (natural death or other forms of removal)
    cm.flow("E_HIV", "S_HIV", args["mu"])  # Death/removal from exposed
    cm.flow("I_HIV", "S_HIV", args["mu"])  # Death/removal from infected
    cm.flow("T_HIV", "S_HIV", args["mu"])  # Death/removal from treatment
    cm.flow("P", "S_HIV", args["mu"])  # Death/removal from protected

    # Basic STI dynamics
    cm.flow(
        "S_STI", "Ia_STI", args["psi"] * beta_STI(y, args) * (y["Ia_STI"] + y["Is_STI"])
    )  # Susceptible to asymptomatic
    cm.flow(
        "S_STI",
        "Is_STI",
        (1 - args["psi"]) * beta_STI(y, args) * (y["Ia_STI"] + y["Is_STI"]),
    )  # Susceptible to symptomatic
    cm.flow(
        "Ia_STI", "S_STI", args["gamma_STI"]
    )  # Asymptomatic to susceptible (recovery)
    cm.flow(
        "Ia_STI", "T_STI", lambda_a(y, args)
    )  # Asymptomatic to tested and treatment
    cm.flow("Is_STI", "T_STI", lambda_a(y, args))  # Symptomatic to tested and treatment
    cm.flow(
        "T_STI", "S_STI", args["gammaT_STI"]
    )  # Treatment to susceptible (immunity loss)

    # Vital dynamics (natural death or other forms of removal)
    cm.flow("Ia_STI", "S_STI", args["mu"])  # Death/removal from asymptomatic
    cm.flow("Is_STI", "S_STI", args["mu"])  # Death/removal from symptomatic
    cm.flow("T_STI", "S_STI", args["mu"])  # Death/removal from treatment

    # Hazard dynamics # TODO: check if implemented correctly
    h, H = icomo.delayed_copy(y["I_HIV"], [y["h"], y["H"]], args["tau"])
    cm._add_dy_to_comp("h", h)
    cm._add_dy_to_comp("H", H)

    # phi_H # TODO: check if implemented correctly
    cm._add_dy_to_comp(
        "phi_H", args["r"] * y["phi_H"] * (1 - y["phi_H"] / args["phi_max"])
    )

    # Return the differential changes
    return cm.dy


# Function to setup the model and return the integrator
def setup_model(args, y0):

    # Define the time span for the simulation
    ts = np.linspace(0, 3600 * 5, 3600)

    # Create an ODE integrator object using the icomo library
    integrator_object = icomo.ODEIntegrator(
        ts_out=ts,  # Output time points
        t_0=min(ts),  # Initial time point
        ts_solver=ts,  # Time points for the solver to use
    )

    # Get the integration function for the model
    integrator = integrator_object.get_func(
        model
    )  # Returns a function that can be used to solve the ODEs defined in the 'model' function

    logger.info("Model setup complete and ready for simulation")

    return integrator
