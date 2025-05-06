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

# Helper functions ----------------------------------------------------------------------------

def fraction2rate(x):
    """
    Converts a fraction to a daily rate.

    This function takes a fraction (or scalar) `x` and converts it into a daily rate
    using the formula `-log(1-x) / 365`. If the input `x` is a scalar, it is expanded
    into a 1D array with four identical elements before the conversion.

    Parameters:
    x (float or array-like): The fraction(s) to be converted.

    Returns:
    jnp.ndarray: A 1D array of daily rates corresponding to the input fraction(s).
    """
    if np.isscalar(x):
        x = jnp.array([x]*4)
    return -jnp.log(1-x) / 365.

def duration2rate(x):
    """
    Converts a duration (in days) to a rate per day.

    If the input is a scalar, it is converted to a 1D array with four identical elements.
    The rate is calculated as the reciprocal of the duration (1 / x) divided by 365,
    effectively converting the duration to a daily rate.

    Parameters:
    x (float or array-like): The duration(s) in days.

    Returns:
    jnp.ndarray: An array of daily rates corresponding to the input durations.
    """
    if np.isscalar(x):
        x = jnp.array([x]*4)
    return 1 / x / 365.



# Parameters ------------------------------------------------------------------------------
N_0 = np.array([0.451, 0.353, 0.125, 0.071])
# HIV params---------------------------------------------
# Parameters we have to decide
# k_on = fraction2rate(0.3)   # annual PrEP uptake rate
# k_off = duration2rate(5.0)  # average duration of taking PrEP per year
# tau_p = fraction2rate(0.95)     # annual ART uptake rate
#k_on = jnp.array([0,0,0,-jnp.log(1-0.5) / 365])   # annual PrEP uptake rate
#k_off = jnp.array([0,0,0, 1 / 5.0 / 365])  # average duration of taking PrEP per year
#tau_p = jnp.array([0,0,0,-jnp.log(1-0.95) / 365])     # annual ART uptake rate

# from GannaRozhnova paper (Elimination prospects of the Dutch HIV epidemic)
c_hiv = jnp.array([0.13, 1.43, 5.44, 18.21]) / 365.0 # per year, average number of partners in risk group l
h = jnp.array([0.62, 0.12, 0.642, 0.0]) # infectivity of untreated individuals in stage k of infection
phis = fraction2rate(0.05)  # per year, annual ART dropout rate
taus = fraction2rate(0.3)   # per year, annual ART uptake rate
gammas = duration2rate( jnp.array([8.21, 54.0, 2.463, 2.737]) )    # per year, rate of transition from stage 1 to 2 for tretaed individuals
rhos = duration2rate ( jnp.array([0.142, 8.439, 1.184, 1.316]) )   # per year, rate of transition from stage 1 to 2 for untreated individuals
mu = 1/45 /365.0 # per year, rate of recruitment to sexually active population
Omega = 1-0.86 # PrEP effectiveness, baseline
epsilon = 0.01 # infectivity of treated individuals
epsilonP = h[0]/2 # infectivity of MSM infected on PrEP
Lambda = jnp.array([0.25]*4) # transmission prob. per partnership
omega = 0.5 # mixing parameter, (0: assortative, 1: proportionate mixing)


#STI params---------------------------------------------
beta_STI = 0.0016 * 7.0  # STI infection rate 
gamma_STI = duration2rate(1.32)  # Recovery rate from asymptomatic STI per day 
gammaT_STI = 1.0 / 7.0  # Recovery rate from treated STI per day [Checked, try with 1/7]
lambda_0 = 1 / 14.0  # Baseline test rate for symptomatic STI 
lambda_P = 2 / 365  # Testing rate due to HIV prevalence 
Psi = 0.85  # Proportion of asymptomatic infections 
m_max = 1  # Maximum value for the exponential modulating factor
m_min = 0.0  # Minimum value for the exponential modulating factor
H_thres = 0.2  # HIV threshold
Sigma = 0.01/365 # Influx
# c = 50 # 1.65 to 203
beta_HIV = 0.6341/365 # HIV infection rate 
sets_of_c = jnp.array([
    [31.0,  40.0,   60.0,  203.0],
    [29.0,  38.0,   73.0,  203.0],
    [14.0,  24.0,  167.0,  203.0],
    [15.0,  40.0,  120.0,  203.0],
    [10.0,  38.0,  141.3, 203.0],
])
c = sets_of_c[2]

#H = 5.0 # HIV hazard
#P = 50.0 # PrEP fraction


args = {
    "N_0": N_0,
    "mu": mu,
    "Omega": Omega,
    "c": c,
    "c_hiv": c_hiv,
    "h": h,
    "epsilon": epsilon,
    "epsilonP": epsilonP,
    "Lambda": Lambda,
    "omega": omega,
    "phis": phis,
    "taus": taus,
    #"tau_p": tau_p,
    "rhos": rhos,
    "gammas": gammas,
    #"k_on": k_on,
    #"k_off": k_off,
    "Psi": Psi,
    "beta_STI": beta_STI,
    "lambda_0": lambda_0,
    "lambda_P": lambda_P,
    "m_max": m_max,
    "m_min": m_min,
    "m_max": m_max,
    "H_thres": H_thres,
    "Sigma": Sigma,
    "beta_STI": beta_STI,
    "gamma_STI": gamma_STI,
    "gammaT_STI": gammaT_STI,
    "beta_HIV": beta_HIV,
    #"H": H,
    #"P": P
    }




# Starting values ----------------------------------------------------------------------------
# Total population stratified by 4 risk groups
N_0 = np.array([0.451, 0.353, 0.125, 0.071])
if (N_0.sum() != 1):
    logger.error("N_0 does not add up to 1.")

# Initial state of the compartments stratified by risk group
y0 = {
    
    # STI starting values
    "S_STI":  0.85 * N_0,    # Susceptible
    "Ia_STI": 0.11 * N_0,    # Infected asymptomatic
    "Is_STI": 0.03 * N_0,    # Infected symptomatic
    "T_STI":  0.01 * N_0,    # Tested and treated

}
all_STI_compartments = ["S_STI", "Ia_STI", "Is_STI", "T_STI"]
if not np.isclose(np.sum(np.array([y0[comp] for comp in all_STI_compartments])), 1):
    logger.error("y_0 does not add up to 1 for STI: %s", np.sum(np.array([y0[comp] for comp in all_STI_compartments])))





def m(args, y):
    """
    Exponential function with three parameters: minimum value, maximum value, and rate/tau.

    Args:
    args (dict): A dictionary containing the parameters 'H', 'm_min', 'm_max', and 'tau_exp'.
    H (float): The current value of 'H'.

    Returns:
    float: The output of the exponential function.
    """
    global logged_exp_logis
    logger.debug("Calculating self-regulation factor factor 'm' using exponential function")

    if not logged_exp_logis:
        logger.info("Using exponential function to calculate m")
        logger.info(
            "Parameters: m_min = %s, m_max = %s",
            args["m_min"],
            args["m_max"],
        )
        logged_exp_logis = True
    res = args["m_min"] + (args["m_max"] - args["m_min"]) * (1 - jnp.exp(-hazard(y, args) / args["H_thres"]))
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
    foi = args["beta_STI"] * (1 - m(args, y)*(1 - prep_fraction(y, args)) ) *contact_matrix(y, args) @ (I_eff/N_0)
    return foi

def contact_matrix(y, args):           
    # this is the matrix named M_ll' in the paper from GannaRozhnova
    mixing = args["omega"] * jnp.tile(args["c_hiv"]*N_0, [4,1]) / jnp.dot(args["c_hiv"], N_0) 
    diagonal = (1-args["omega"])*jnp.identity(4)
    contact_matrix = mixing + diagonal

    # TODO delete this later, just for testing
    #contact_matrix = jnp.ones((4,4))  # Initialize a 4x4 matrix with zeros

    # Normalize by max eigenvalue
    eigvals = jnp.linalg.eigvals(contact_matrix)
    lambda_max = jnp.max(jnp.real(eigvals))  # Ensure we take the real part in case of complex values
    contact_matrix_normalized = contact_matrix / lambda_max


    return contact_matrix_normalized

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
    return args["lambda_0"] + lambda_a(y, args)



def main_model(t, y, args):
    cm = icomo.CompModel(y)  # Initialize the compartmental model


    # STI dynamics-------------------------------------------------------------------------------------------------------------------------------------------------
    cm.flow("S_STI",  "Ia_STI", (args["Psi"])   * foi_STI(y, args))     # Susceptible to asymptomatic
    cm.flow("S_STI",  "Is_STI", (1-args["Psi"]) * foi_STI(y, args))     # Susceptible to symptomatic
    cm.flow("Ia_STI", "S_STI",  args["gamma_STI"])                      # Asymptomatic to susceptible (recovery)
    cm.flow("Ia_STI", "T_STI",  lambda_a(y,args))                       # Asymptomatic to tested and treatment
    cm.flow("Is_STI", "T_STI",  lambda_s(y,args))                       # Symptomatic to tested and treatment
    cm.flow("T_STI",  "S_STI",  args["gammaT_STI"])                     # Treatment to susceptible (immunity loss)
    cm.dy["S_STI"] -= args["Sigma"]                                     # Influx
    cm.dy["Ia_STI"] += args["Psi"] * args["Sigma"]                      # Influx
    cm.dy["Is_STI"] += (1-args["Psi"]) * args["Sigma"]                  # Influx
    # Vital dynamics
    cm.flow("Ia_STI", "S_STI", args["mu"])  # Death/removal from asymptomatic
    cm.flow("Is_STI", "S_STI", args["mu"])  # Death/removal from symptomatic
    cm.flow("T_STI", "S_STI", args["mu"])  # Death/removal from treatment





    return cm.dy

