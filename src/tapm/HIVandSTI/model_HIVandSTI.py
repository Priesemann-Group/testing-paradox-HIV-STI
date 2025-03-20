# Parameters from GannaRozhnova's paper (Elimination prospects of the Dutch HIV epidemic among men who have sex with men in the era of
#  pre-exposure prophylaxis)  https://pubmed.ncbi.nlm.nih.gov/30379687/




import numpy as np 
import icomo
import jax.numpy as jnp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to track logging
logged_exp_logis = False



# Total population stratified by 4 risk groups
N_0 = np.array([0.451, 0.353, 0.125, 0.071])
if (N_0.sum() != 1):
    logger.error("N_0 does not add up to 1.")

# Initial state of the compartments stratified by risk group
y0 = {
    # HIV compartments
    "S":  [0.49874, 0.4919, 0.42845, 0.42395] * N_0,    # Susceptible
    "SP": [0.49874, 0.4919, 0.42845, 0.42395] * N_0,    # Susceptible on PrEP
    "I1": [0.00001, 0.0001, 0.001,   0.001]   * N_0,    # Infected in stage 1
    "IP": [0.00001, 0.0001, 0.0001,  0.0001]  * N_0,    # Infected in stage 1 on PrEP
    "I2": [0.001,   0.001,  0.01,    0.01]    * N_0,    # Infected in stage 2
    "I3": [0.0001,  0.001,  0.01,    0.01]    * N_0,    # Infected in stage 3
    "I4": [0.0001,  0.001,  0.001,   0.01]    * N_0,    # Infected in stage 4
    "A1": [0.0001,  0.001,  0.001,   0.01]    * N_0,    # Infected in stage 1 on ART
    "A2": [0.001,   0.01,   0.1,     0.1]     * N_0,    # Infected in stage 2 on ART
    "A3": [0.0001,  0.001,  0.01,    0.01]    * N_0,    # Infected in stage 3 on ART
    "A4": [0.0001,  0.001,  0.01,    0.001]   * N_0,    # Infected in stage 4 on ART
    "D":  [0.0,     0.0,    0.0,     0.0]     * N_0,    # Deceased from HIV
    
    # STI starting values
    "S_STI":  0.65 * N_0,    # Susceptible
    "Ia_STI": 0.15 * N_0,    # Infected asymptomatic
    "Is_STI": 0.15 * N_0,    # Infected symptomatic
    "T_STI":  0.05 * N_0,    # Tested and treated

    # Hazard
    "H": [0.0, 0.0, 0.0, 0.0] * N_0, # hazard
    # maybe this needs three compartments instead of two
    # also not sure about the initial values
}
# if not all(np.isclose(np.array([x for x in y0.values()]).sum(axis=0), 2*N_0)):
#     logger.error("y_0 does not add up to N_0.")



# TODO: check if the interpretation as fraction/duration 2 rate is correct

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


# Parameters we have to decide
k_on = fraction2rate(0.3)        # annual PrEP uptake rate
k_off = duration2rate(5.0)        # average duration of taking PrEP per year

tau_p = fraction2rate(0.95)     # annual ART uptake rate

c = jnp.array([0.13, 1.43, 5.44, 18.21]) / 365.0 # per year, average number of partners in risk group l
h = jnp.array([0.62, 0.12, 0.642, 0.0])         # infectivity of untreated individuals in stage k of infection

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




##---------------------------------------------
m_function = "exponential",  # Modulating function
beta_HIV = 0.6341 / 365.0 # HIV infection rate per day
beta_STI = 0.0016 * 7.0  # STI infection rate [Checked]
#mu = 1.0 / 45.0 / 365.0,  # Natural death rate per day [Checked]
gamma_STI = 1.0 / 1.32 / 365.0  # Recovery rate from asymptomatic STI per day [Checked]
gammaT_STI = 1.0 / 7.0  # Recovery rate from treated STI per day [Checked, try with 1/7]
lambda_0 = 1 / 14.0  # Baseline test rate for symptomatic STI [Checked]
lambda_P = 2 / 365  # Testing rate due to HIV prevalence [Checked]
asymptomatic = 0.85  # Proportion of asymptomatic infections [Checked]
m_max = 0.8  # Maximum modulating factor
H_thres = 0.1  # HIV threshold
scaling_factor_m_eps = 1.0  # Scaling factor for the exponential modulating factor
m_eps = 0.01  # Small constant for smoothing
#Phi_r = 40.0  # Not used in the current model
#H_tau = 20.0  # Not used in the current model
contact = 50.0  # Scaling factor for HIV interaction term
#H = 0.1  # Initial HIV prevalence
#P_HIV = 0.25  # Initial proportion of HIV positive individuals
min_exp = 0.0  # Minimum value for the exponential modulating factor
max_exp = 1.0  # Maximum value for the exponential modulating factor
tau_exp = 0.2  # Time constant for the exponential modulating factor
Sigma = 0.01/365 # Influx
delay = jnp.array([20.]) # Delay for the Hazard
##---------------------------------------------


args = {
    "beta_HIV": beta_HIV,
    "N_0": N_0,
    "mu": mu,
    "Omega": Omega,
    "c": c,
    "h": h,
    "epsilon": epsilon,
    "epsilonP": epsilonP,
    "Lambda": Lambda,
    "omega": omega,
    "phis": phis,
    "taus": taus,
    "tau_p": tau_p,
    "rhos": rhos,
    "gammas": gammas,
    "k_on": k_on,
    "k_off": k_off,
    "asymptomatic": asymptomatic,
    "beta_STI": beta_STI,
    "lambda_0": lambda_0,
    "lambda_P": lambda_P,
    "m_function": m_function,
    "min_exp": min_exp,
    "max_exp": max_exp,
    "tau_exp": tau_exp,
    "m_eps": m_eps,
    "Sigma": Sigma,
    "scaling_factor_m_eps": scaling_factor_m_eps,
    "gamma_STI": gamma_STI,
    "gammaT_STI": gammaT_STI,
    "contact": contact,
    "delay": delay,
    }




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to track logging
logged_exp_logis = False
logged_tau = False


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
    #H = jnp.array(H[-1]) # (hazard is only last compartment of H)
    # min_exp = args["min_exp"]
    # max_exp = args["max_exp"]
    # tau_exp = args["tau_exp"] * args["scaling_factor_m_eps"]

    if not logged_exp_logis:
        logger.info("Using exponential function to calculate m")
        logger.info(
            "Parameters: min_exp = %s, max_exp = %s, tau_exp = %s",
            args["min_exp"],
            args["max_exp"],
            args["tau_exp"] * args["scaling_factor_m_eps"],
        )
        logged_exp_logis = True

    return args["min_exp"] + (args["max_exp"] - args["min_exp"]) * (1 - jnp.exp(-hazard(y, args) / (args["tau_exp"] * args["scaling_factor_m_eps"])))


def foi_HIV(y, args):
    """
    Computes the force of infection for HIV.
    Args:
        y (dict): A dictionary containing the state variables.
        args (dict): A dictionary containing the model parameters. Expected keys include:

    Returns:
        ndarray: The computed force of infection as a vector.
    """

    I_eff = args["epsilonP"]*y["IP"] \
          + jnp.dot(args["h"], jnp.array([y["I1"], y["I2"], y["I3"], y["I4"]])) \
          + args["epsilon"]*(y["A1"] + y["A2"] + y["A3"] + y["A4"])
    foi = args["Lambda"] * args["c"] * contact_matrix(args) @ I_eff

    return foi


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
    foi = (1 - m(args, y)*(1 - prep_fraction(y)) ) * contact_matrix(args) @ I_eff
    
    return foi



def contact_matrix(args):           # first implementation, TODO: make sure it is correctly orientated

    mixing = args["omega"] * jnp.tile(args["c"]*args["N_0"], [4,1]) / jnp.dot(args["c"], args["N_0"])
    diagonal = (1-args["omega"])*jnp.identity(4)

    return mixing + diagonal


def hazard(y, args):
    """
    Calculates the hazard from the HIV model used for risk-perception.
    Right now the hazard is just the fraction of people on ART in the HIV model.

    Returns:
        Value as float64.
    """
    #return 0.02    # debuging, remove later [TODO]
    return jnp.sum( jnp.array([y["A1"], y["A2"], y["A3"], y["A4"]]) )

def prep_fraction(y):
    """
    Calculates fraction of people on PrEP in the HIV model.

    Returns:
        Value as float64.
    """
    #return 0.2    # debuging, remove later [TODO]
    return jnp.sum( jnp.array([y["SP"], y["IP"]]) )

def lambda_a(y, args):
    """
    Calculates STI testing rate for asymtomatic infected.

    Returns:
        Array of rates with dimension [1, risk groups].
    """
    risk_induced_testing = args["contact"] * (1 - m(args, y)) * args["beta_HIV"] * hazard(y, args) * (1 - prep_fraction(y))
    prep_induced_testing = args["lambda_P"] * prep_fraction(y)
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
    
    # HIV dynamics-------------------------------------------------------------------------------------------------------------------------------
    cm.flow("S", "I1", foi_HIV(y, args))                       # check later
    cm.flow("SP", "IP", args["Omega"]*foi_HIV(y, args))        # probably a prefactor
    cm.flow("IP", "A1", args["tau_p"])
    cm.flow("S", "SP", args["k_on"])
    cm.flow("SP", "S", args["k_off"])


    def add_flows(starts, ends, rates):
        for i in range(len(starts)):
            cm.flow(starts[i], ends[i], rates[i])

    Is = ["I1", "I2", "I3", "I4", "D"]
    As = ["A1", "A2", "A3", "A4", "D"]

    add_flows(Is[:-1], Is[1:], args["rhos"])
    add_flows(As[:-1], As[1:], args["gammas"])

    add_flows(Is[:-1], As[:-1], args["taus"])
    add_flows(As[:-1], Is[:-1], args["phis"])


    # STI dynamics-------------------------------------------------------------------------------------------------------------------------------------------------
    cm.flow("S_STI",  "Ia_STI", (args["asymptomatic"])   * foi_STI(y, args))     # Susceptible to asymptomatic
    cm.flow("S_STI",  "Is_STI", (1-args["asymptomatic"]) * foi_STI(y, args))     # Susceptible to symptomatic
    cm.flow("Ia_STI", "S_STI",  args["gamma_STI"])                               # Asymptomatic to susceptible (recovery)
    cm.flow("Ia_STI", "T_STI",  lambda_a(y,args))                                # Asymptomatic to tested and treatment
    cm.flow("Is_STI", "T_STI",  lambda_s(y,args))                                # Symptomatic to tested and treatment
    cm.flow("T_STI",  "S_STI",  args["gammaT_STI"])                              # Treatment to susceptible (immunity loss)


    # Vital dynamics-----------------------------------------------------------------------------------------------------------------------------------------------
    for comp in cm.dy.keys():
        if comp not in ["D", "H"]:
            cm.dy[comp] -= mu * y[comp]
    for comp in ["S", "S_STI"]:
        cm.dy[comp] += mu*args["N_0"]


    ### not used right now:
    # hazard--------------------------------------------------------------------------------------------------------------------------------------------------------
    # def hazard_flow(start_comp, end_comp, rate):
    #     for i in range(len(start_comp)):
    #         cm.delayed_copy(start_comp[i], end_comp[i], rate[i])
    # compartments = ["A1", "A2", "A3", "A4"]

    # hazard_flow(compartments, "H", args["delay"])
    # #tau_delay = jnp.array([tau for _ in compartments]) # Create an array of delays with the same length as compartments

    # for i in range(len(compartments)):
    #     # # y[start_comp] = jnp.array(y[start_comp])
    #     # if not isinstance(y[start_comp], jnp.ndarray):
    #     #     raise ValueError(f"{start_comp} must be an array-like object")
    #     cm.delayed_copy(comp_to_copy=compartments[i], delayed_comp="H", tau_delay=delay)
    # # temp = ["A11","A12","A13","A14","A21","A22","A23","A24","A31","A32","A33","A34","A41","A42","A43","A44"]
    # # cm.delayed_copy(temp,"H", jnp.array([tau, tau]))
    # #cm.view_graph()

    return cm.dy

