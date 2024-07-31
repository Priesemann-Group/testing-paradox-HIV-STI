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
import icomo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to calculate the modulating factor 'm' based on provided arguments
def m(args):
    logger.debug("Calculating modulating factor 'm'")
    return args["m_max"] - args["m_max"] / args["H_thres"] * args[
        "m_eps"
    ] * jax.numpy.log(1 + jax.numpy.exp((args["H_thres"] - args["H"]) / args["m_eps"]))


# Function to calculate the testing rate of STI
def lambda_STI(args):
    logger.debug("Calculating testing rate of STI")
    return (
        args["lambda_0_a"]  # Baseline test rate
        + args["c"]
        * (1 - m(args))
        * args["beta_HIV"]
        * args["H"]
        * (1 - args["P_HIV"])  # HIV dependent term
        + args["lambda_P"]
        * args["P_HIV"]  # Proportional infection rate due to HIV prevalence
    )


# Function to calculate infection from asymptomatic STI individuals
def infect_ia(y, args):
    logger.debug("Calculating infection from asymptomatic STI individuals")
    return (
        (args["asymptomatic"])
        * (1 - m(args) * (1 - args["P_HIV"]))
        * args["beta_STI"]
        * (y["Ia_STI"] + y["Is_STI"])
    )


# Function to calculate infection from symptomatic STI individuals
def infect_is(y, args):
    logger.debug("Calculating infection from symptomatic STI individuals")
    return (
        (1 - args["asymptomatic"])
        * (1 - m(args) * (1 - args["P_HIV"]))
        * args["beta_STI"]
        * (y["Is_STI"] + y["Ia_STI"])
    )


# Main model function that defines the differential equations of the system
def model(t, y, args):
    logger.debug("Defining the differential equations of the system")
    cm = icomo.CompModel(y)  # Initialize the compartmental model

    # Basic STI dynamics
    cm.flow("S_STI", "Ia_STI", infect_ia(y, args))  # Susceptible to asymptomatic
    cm.flow("S_STI", "Is_STI", infect_is(y, args))  # Susceptible to symptomatic
    cm.flow(
        "Ia_STI", "S_STI", args["gamma_STI"]
    )  # Asymptomatic to susceptible (recovery)
    cm.flow("Ia_STI", "T_STI", lambda_STI(args))  # Asymptomatic to tested and treatment
    cm.flow("Is_STI", "T_STI", args["lambda_0"])  # Symptomatic to tested and treatment
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


# If this script is run directly, setup the model
# if __name__ == "__main__":
#     integrator, y0, args = setup_model()
