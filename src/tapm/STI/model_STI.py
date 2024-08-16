# This python file contains the model definition for the STI dynamics in the model.
# The model is defined as a set of differential equations that describe the flow of individuals between different compartments.
# The model includes compartments for susceptible individuals, asymptomatic STI cases, symptomatic STI cases, and individuals undergoing treatment for STI.
# The model also includes parameters that describe the transmission dynamics of the STI, the testing and treatment rates, and the impact of HIV on STI transmission.
# The model is implemented using the icomo library, which provides tools for defining and solving compartmental models. The model is set up to be integrated over a time span of 5 years, with output time points every hour.
# The model is ready for simulation once it is set up.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import jax
import jax.numpy as jnp
import icomo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to track logging
logged_exp_logis = False


# Function to calculate the modulating factor 'm' based on provided arguments
def m_logistic(args, H=None):
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
            "Parameters: m_max = %s, H_thres = %s, m_eps = %s",
            args["m_max"],
            args["H_thres"],
            args["m_eps"],
        )
        logged_exp_logis = True
    if H is None:
        H = args["H"]
    return args["m_max"] - args["m_max"] / args["H_thres"] * args[
        "m_eps"
    ] * jax.numpy.log(1 + jax.numpy.exp((args["H_thres"] - H) / args["m_eps"]))


def m_exponential(args, H=None):
    """
    Exponential function with three parameters: minimum value, maximum value, and rate/tau.

    Args:
    args (dict): A dictionary containing the parameters 'H', 'min_exp', 'max_exp', and 'tau_exp'.
    H (float): The current value of 'H'.

    Returns:
    float: The output of the exponential function.
    """
    global logged_exp_logis
    logger.debug(
        "Calculating self-regulation factor factor 'm' using exponential function"
    )
    if H is None:
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


def m(args, H):
    """
    Select the appropriate function to calculate the self-regulation factor 'm' based on the arguments.

    Args:
        args (dict): A dictionary containing the arguments for the calculation.

    Returns:
        The result of the calculation.
    """
    if args["m_function"] == "exponential":
        return m_exponential(args, H)
    elif args["m_function"] == "logistic":
        return m_logistic(args, H)
    else:
        raise ValueError("Invalid m_function specified in args")


# Function to calculate the testing rate of STI
def lambda_a(args):
    """
    Calculate the testing rate of assymptomatic STI.

    Args:
        args (dict): A dictionary containing the following parameters:
            - lambda_0 (float): Baseline test rate
            - c (float): Constant term
            - m (function): Function to calculate m value
            - beta_HIV (float): HIV transmission rate
            - H (float): Number of susceptible individuals
            - P_HIV (float): HIV prevalence

    Returns:
        float: The testing rate of STI.

    """
    logger.debug("Calculating testing rate of STI")
    return (
        args["lambda_0"]  # Baseline test rate
        + args["c"]
        * (1 - m(args, H=None))
        * args["beta_HIV"]
        * args["H"]
        * (1 - args["P_HIV"])  # HIV dependent term
        + args["lambda_P"]
        * args["P_HIV"]  # Proportional infection rate due to HIV prevalence
    )


# Function to calculate infection from asymptomatic STI individuals
def infect_ia(y, args):
    """
    Calculates the infection from asymptomatic STI individuals.

    Parameters:
    - y: A dictionary containing the current state variables.
    - args: A dictionary containing the model parameters.

    Returns:
    - The calculated infection from asymptomatic STI individuals.
    """
    logger.debug("Calculating infection from asymptomatic STI individuals")
    return (
        (args["asymptomatic"])
        * (1 - m(args, H=None) * (1 - args["P_HIV"]))
        * args["beta_STI"]
        * (y["Ia_STI"] + y["Is_STI"])
    )


# Function to calculate infection from symptomatic STI individuals
def infect_is(y, args):
    """
    Calculate the infection from symptomatic STI individuals.

    Parameters:
    - y: A dictionary containing the number of individuals in different STI compartments.
    - args: A dictionary containing the model parameters.

    Returns:
    - The calculated infection from symptomatic STI individuals.
    """
    logger.debug("Calculating infection from symptomatic STI individuals")
    return (
        (1 - args["asymptomatic"])
        * (1 - m(args, H=None) * (1 - args["P_HIV"]))
        * args["beta_STI"]
        * (y["Is_STI"] + y["Ia_STI"])
    )


# Main model function that defines the differential equations of the system
def model(t, y, args):
    """
    Calculate the differential changes in a compartmental model for STI dynamics.

    Parameters:
    t (float): The time at which the model is evaluated.
    y (list): The current values of the compartments in the model.
    args (dict): Additional arguments required for the model calculations.

    Returns:
    list: The differential changes in the compartments.

    """
    logger.debug("Defining the differential equations of the system")
    cm = icomo.CompModel(y)  # Initialize the compartmental model

    # Basic STI dynamics
    cm.flow("S_STI", "Ia_STI", infect_ia(y, args))  # Susceptible to asymptomatic
    cm.flow("S_STI", "Is_STI", infect_is(y, args))  # Susceptible to symptomatic
    cm.flow(
        "Ia_STI", "S_STI", args["gamma_STI"]
    )  # Asymptomatic to susceptible (recovery)
    cm.flow("Ia_STI", "T_STI", lambda_a(args))  # Asymptomatic to tested and treatment
    cm.flow("Is_STI", "T_STI", args["lambda_s"])  # Symptomatic to tested and treatment
    cm.flow(
        "T_STI", "S_STI", args["gammaT_STI"]
    )  # Treatment to susceptible (immunity loss)

    # Vital dynamics (natural death or other forms of removal)
    cm.flow("Ia_STI", "S_STI", args["mu"])  # Death/removal from asymptomatic
    cm.flow("Is_STI", "S_STI", args["mu"])  # Death/removal from symptomatic
    cm.flow("T_STI", "S_STI", args["mu"])  # Death/removal from treatment

    # Return the differential changes
    return cm.dy


# Function to setup the model and return the integrator
def setup_model(args, y0):
    """
    Set up the model for simulation.

    Args:
        args: Additional arguments for setting up the model.
        y0: Initial conditions for the model.

    Returns:
        integrator: A function that can be used to solve the ODEs defined in the 'model' function.
    """

    # Define the time span for the simulation
    ts = np.linspace(0, 3600 * 10, 3600)

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


# If this script is run directly, setup the model
# if __name__ == "__main__":
#     integrator, y0, args = setup_model()
