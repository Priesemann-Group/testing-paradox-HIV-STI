import numpy as np 
import icomo
import jax.numpy as jnp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to track logging
logged_exp_logis = False


# Parameters we have to decide
PrEPuptake_rg1 = 0.3 # annual PrEP uptake in risk group 1 (fraction)
PrEPuptake_rg2 = 0.3 # annual PrEP uptake in risk group 2 (fraction)
PrEPuptake_rg3 = 0.3 # annual PrEP uptake in risk group 3 (fraction)
PrEPuptake_rg4 = 0.3 # annual PrEP uptake in risk group 4 (fraction)


# Parameters from GannaRozhnova's paper (Elimination prospects of the Dutch HIV epidemic among men who have sex with men in the era of pre-exposure prophylaxis)
N0 = 210000 # initial population size
N01 = 0.451*N0 # initial population size of risk group 1
N02 = 0.353*N0 # initial population size of risk group 2
N03 = 0.125*N0 # initial population size of risk group 3
N04 = 0.071*N0 # initial population size of risk group 4
N0s = [N01,N02,N03,N04]
mu = 1/45 /360.0 # per year, rate of recruitment to sexually active population
Omega = 1-0.86 # PrEP effectiveness, baseline
c = [0.13/360.0, 1.43/360.0, 5.44/360.0, 18.21/360.0] # per year, average number of partners in risk group l
h = [0.62,0.12,0.642,0.0] # infectivity of untreated individuals in stage k of infection
epsilon = 0.01 # infectivity of treated individuals
epsilonP = h[0]/2 # infectivity of MSM infected on PrEP
Lambda = 0.25 # transmission prob. per partnership
omega = 0.5 # mixing parameter, (0: assortative, 1: proportionate mixing)
Phi = -jnp.log(1-0.05) / 360.0 # per year, annual ART dropout rate
tau = -jnp.log(1-0.3) / 360.0 # per year, annual ART uptake rate
tauP1 = -jnp.log(1-0.95) / 360.0 # per year, annual ART uptake rate for MSM infected on PrEP
tauP2 = tauP1
tauP3 = tauP1
tauP4 = tauP1
tauPs = [tauP1,tauP2,tauP3,tauP4]
rho1 = 1/0.142 / 360.0 # per year, rate of transition from stage 1 to 2 for untreated individuals
rho2 = 1/8.439 / 360.0 # per year, rate of transition from stage 2 to 3 for untreated individuals
rho3 = 1/1.184 / 360.0 # per year, rate of transition from stage 3 to 4 for untreated individuals
rho4 = 1/1.316 / 360.0 # per year, mortality rate for untreated individuals
rhos = [rho1,rho2,rho3,rho4]
gamma1 = 1/8.21 / 360.0 # per year, rate of transition from stage 1 to 2 for tretaed individuals
gamma2 = 1/54.0 / 360.0 # per year, rate of transition from stage 2 to 3 for treated individuals
gamma3 = 1/2.463 / 360.0 # per year, rate of transition from stage 3 to 4 for treated individuals
gamma4 = 1/2.737 / 360.0 # per year, mortality rate for treated individuals
gammas = [gamma1,gamma2,gamma3,gamma4]
Kon1 = -jnp.log(1-PrEPuptake_rg1) / 360.0 # annual PrEP uptake rate in risk group 1
Kon2 = -jnp.log(1-PrEPuptake_rg2) / 360.0 # annual PrEP uptake rate in risk group 2
Kon3 = -jnp.log(1-PrEPuptake_rg3) / 360.0 # annual PrEP uptake rate in risk group 3
Kon4 = -jnp.log(1-PrEPuptake_rg4) / 360.0 # annual PrEP uptake rate in risk group 4
Kons = [Kon1,Kon2,Kon3,Kon4]
Koff1 = 1/5.0 / 360.0 # per year, average duration of taing PrEP in risk group 1
Koff2 = Koff1 
Koff3 = Koff1 
Koff4 = Koff1 
Koffs = [Koff1,Koff2,Koff3,Koff4]


##---------------------------------------------
m_function = "exponential",  # Modulating function
#beta_HIV = 0.6341 / 360.0 # HIV infection rate per day
beta_STI = 0.0016 * 7.0  # STI infection rate [Checked]
#mu = 1.0 / 45.0 / 360.0,  # Natural death rate per day [Checked]
gamma_STI = 1.0 / 1.32 / 360.0  # Recovery rate from asymptomatic STI per day [Checked]
gammaT_STI = 1.0 / 7.0  # Recovery rate from treated STI per day [Checked, try with 1/7]
lambda_0 = 0.0  # Baseline test rate for asymptomatic STI [Checked]
lambda_s = 1 / 14.0  # Baseline test rate for symptomatic STI [Checked]
lambda_P = 2 / 360  # Testing rate due to HIV prevalence [Checked]
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
delay = 200 # Delay for the Hazard
##---------------------------------------------

# Initial state of the compartments
y0 = {
    # Susceptible per risk group
    "S1": 0.49874 * N01,
    "S2": 0.49185 * N02,
    "S3": 0.47345 * N03,
    "S4": 0.42395 * N04,
    
    # Susceptible on PrEP per risk group
    "SP1": 0.49874 * N01,
    "SP2": 0.49185 * N02,
    "SP3": 0.47345 * N03,
    "SP4": 0.42395 * N04,
    
    # Infected in stage 1 per risk group
    "I11": 0.00001 * N01,
    "I21": 0.0001 * N02,
    "I31": 0.001 * N03,
    "I41": 0.001 * N04,
    
    # Infected on PrEP in stage 1 per risk group
    "IP11": 0.00001 * N01,
    "IP21": 0.0001 * N02,
    "IP31": 0.0001 * N03,
    "IP41": 0.0001 * N04,
    
    # Infected in stage 2 per risk group
    "I12": 0.001 * N01,
    "I22": 0.001 * N02,
    "I32": 0.01 * N03,
    "I42": 0.01 * N04,
    
    # Infected in stage 3 per risk group
    "I13": 0.0001 * N01,
    "I23": 0.001 * N02,
    "I33": 0.01 * N03,
    "I43": 0.01 * N04,
    
    # Infected in stage 4 per risk group
    "I14": 0.0001 * N01,
    "I24": 0.001 * N02,
    "I34": 0.001 * N03,
    "I44": 0.01 * N04,
    
    # ART in stage 1 per risk group
    "A11": 0.0001 * N01,
    "A21": 0.001 * N02,
    "A31": 0.001 * N03,
    "A41": 0.01 * N04,
    
    # ART in stage 2 per risk group
    "A12": 0.001 * N01,
    "A22": 0.01 * N02,
    "A32": 0.1 * N03,
    "A42": 0.1 * N04,
    
    # ART in stage 3 per risk group
    "A13": 0.0001 * N01,
    "A23": 0.001 * N02,
    "A33": 0.01 * N03,
    "A43": 0.01 * N04,
    
    # ART in stage 4 per risk group
    "A14": 0.0001 * N01,
    "A24": 0.001 * N02,
    "A34": 0.01 * N03,
    "A44": 0.001 * N04,

    #"H": jnp.array([0.0, 0.0]), # hazard
    # maybe this needs three compartments instead of two
    # also not sure about the initial values
    
    # STI starting values
    "S1_STI": 0.65 * N01,
    "S2_STI": 0.65 * N02,
    "S3_STI": 0.65 * N03,
    "S4_STI": 0.65 * N04,
    "Ia1_STI": 0.15 * N01,
    "Ia2_STI": 0.15 * N02,
    "Ia3_STI": 0.15 * N03,
    "Ia4_STI": 0.15 * N04,
    "Is1_STI": 0.15 * N01,
    "Is2_STI": 0.15 * N02,
    "Is3_STI": 0.15 * N03,
    "Is4_STI": 0.15 * N04,
    "T1_STI": 0.05 * N01,
    "T2_STI": 0.05 * N02,
    "T3_STI": 0.05 * N03,
    "T4_STI": 0.05 * N04,
}


args = dict(N0s=N0s, mu=mu, Omega=Omega, c=c, h=h, epsilon=epsilon, epsilonP=epsilonP, Lambda=Lambda, omega=omega, Phi=Phi, tau = tau, tauPs=tauPs, rhos=rhos, gammas=gammas, Kons=Kons, Koffs=Koffs, asymptomatic=asymptomatic, beta_STI=beta_STI, lambda_0=lambda_0, lambda_P=lambda_P, lambda_s=lambda_s, m_function=m_function, min_exp=min_exp, max_exp=max_exp, tau_exp=tau_exp, m_eps=m_eps, Sigma=Sigma, scaling_factor_m_eps=scaling_factor_m_eps, gamma_STI=gamma_STI, gammaT_STI=gammaT_STI, contact=contact, delay=delay)

#checked
def calculate_N(y): # number of people per risk group for a given state y (which means at a given time) for HIV
    N1 = jnp.sum(jnp.array([y["S1"],y["SP1"],y["I11"],y["IP11"],y["I12"],y["I13"],y["I14"],y["A11"],y["A12"],y["A13"],y["A14"]]))
    N2 = jnp.sum(jnp.array([y["S2"],y["SP2"],y["I21"],y["IP21"],y["I22"],y["I23"],y["I24"],y["A21"],y["A22"],y["A23"],y["A24"]]))
    N3 = jnp.sum(jnp.array([y["S3"],y["SP3"],y["I31"],y["IP31"],y["I32"],y["I33"],y["I34"],y["A31"],y["A32"],y["A33"],y["A34"]]))
    N4 = jnp.sum(jnp.array([y["S4"],y["SP4"],y["I41"],y["IP41"],y["I42"],y["I43"],y["I44"],y["A41"],y["A42"],y["A43"],y["A44"]]))
    return jnp.array([N1,N2,N3,N4])

#checked
def M(l1,l2,y,args): # mixing matrix
    c = args["c"] # unpack args we need
    omega = args["omega"]
    l1 = l1-1 # because we start counting from 0
    l2 = l2-1
    Ns = calculate_N(y)
    c_array = jnp.array(c)
    return omega*c_array[l2]*Ns[l2]/(jnp.sum(jnp.multiply(c_array,Ns))) + (1-omega)*jnp.where(l1 == l2, 1, 0)

# checked
def JP(l,y,args): # force of infection per year in group l
    h = jnp.array(args["h"]) # unpack args we need
    epsilon = args["epsilon"]
    epsilonP = args["epsilonP"]
    Lambda = args["Lambda"]
    c = jnp.array(args["c"])
    Is_all = jnp.array([[y["I11"], y["I12"], y["I13"], y["I14"]],[y["I21"], y["I22"], y["I23"], y["I24"]],[y["I31"], y["I32"], y["I33"], y["I34"]],[y["I41"], y["I42"], y["I43"], y["I44"]]])
    Is_l = Is_all[l-1] # l-1 because we start counting from 0
    IPs_l1 = jnp.array([y["IP11"],y["IP21"],y["IP31"],y["IP41"]])
    N = calculate_N(y)
    As_all = jnp.array([[y["A11"], y["A12"], y["A13"], y["A14"]],[y["A21"], y["A22"], y["A23"], y["A24"]],[y["A31"], y["A32"], y["A33"], y["A34"]],[y["A41"], y["A42"], y["A43"], y["A44"]]])
    As_l = As_all[l-1]
    Ms_l = jnp.array([M(l,1,y,args),M(l,2,y,args),M(l,3,y,args),M(l,4,y,args)])
    
    innersums = jnp.array([jnp.sum(h*Is_all[ll]/N[ll] + epsilon*As_all[ll]/N[ll]) for ll in range(4)]) # sum over k for different l
    sum = jnp.sum(Ms_l*(jnp.add(epsilonP*IPs_l1/N,innersums))) # sum over l'
    JP = Lambda * c[l-1] * sum
    return JP

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
    min_exp = args["min_exp"]
    max_exp = args["max_exp"]
    tau_exp = args["tau_exp"] * args["scaling_factor_m_eps"]

    if not logged_exp_logis:
        logger.info("Using exponential function to calculate m")
        logger.info(
            "Parameters: min_exp = %s, max_exp = %s, tau_exp = %s",
            min_exp,
            max_exp,
            tau_exp,
        )
        logged_exp_logis = True

    compartments = ["A11","A12","A13","A14","A21","A22","A23","A24","A31","A32","A33","A34","A41","A42","A43","A44"]
    compartment_values = jnp.array([y[key] for key in compartments])
    hazard = jnp.sum(compartment_values) / jnp.sum(calculate_N(y))
    return min_exp + (max_exp - min_exp) * (1 - jnp.exp(-hazard / tau_exp))


def Ml1_first_term(y, args):
    c = args["c"] # unpack args we need
    omega = args["omega"]
    Ns = jnp.array(calculate_N(y))
    Ns_array = jnp.array(Ns)
    c_array = jnp.array(c)
    return omega*c_array[0]*Ns_array[0]/(jnp.sum(c_array*Ns_array, axis=0))

def Ml2_first_term(y, args):
    c = args["c"] # unpack args we need
    omega = args["omega"]
    Ns = jnp.array(calculate_N(y))
    Ns_array = jnp.array(Ns)
    c_array = jnp.array(c)
    return omega*c_array[1]*Ns_array[1]/(jnp.sum(c_array*Ns_array, axis=0))

def Ml3_first_term(y, args):
    c = args["c"] # unpack args we need
    omega = args["omega"]
    Ns = jnp.array(calculate_N(y))
    Ns_array = jnp.array(Ns)
    c_array = jnp.array(c)
    return omega*c_array[2]*Ns_array[2]/(jnp.sum(c_array*Ns_array, axis=0))

def Ml4_first_term(y, args):
    c = args["c"] # unpack args we need
    omega = args["omega"]
    Ns = jnp.array(calculate_N(y))
    Ns_array = jnp.array(Ns)
    c_array = jnp.array(c)
    return omega*c_array[3]*Ns_array[3]/(jnp.sum(c_array*Ns_array, axis=0))

def J1(y, args):
    beta_0_sti = jnp.array(args["beta_STI"])
    c = args["c"]
    M_second_terms = jnp.array([1-omega,0,0,0])
    Is = jnp.array([y["Is1_STI"],y["Is2_STI"],y["Is3_STI"],y["Is4_STI"]])
    Ia = jnp.array([y["Ia1_STI"],y["Ia2_STI"],y["Ia3_STI"],y["Ia4_STI"]])
    N = calculate_N(y)
    Ms = jnp.array([Ml1_first_term(y, args),Ml2_first_term(y, args),Ml3_first_term(y, args),Ml4_first_term(y, args)])
    innersums = [jnp.sum(beta_0_sti*(Is[ll]+Ia[ll])/N[ll]) for ll in range(4)]
    innersums_array = jnp.array(innersums)
    sum = jnp.sum((jnp.add(Ms,M_second_terms)*(innersums_array)))
    JP = Lambda * c[0] * sum
    return JP

def J2(y, args):
    beta_0_sti = jnp.array(args["beta_STI"])
    c = args["c"]
    M_second_terms = jnp.array([0,1-omega,0,0])
    Is = jnp.array([y["Is1_STI"],y["Is2_STI"],y["Is3_STI"],y["Is4_STI"]])
    Ia = jnp.array([y["Ia1_STI"],y["Ia2_STI"],y["Ia3_STI"],y["Ia4_STI"]])
    N = calculate_N(y)
    Ms = jnp.array([Ml1_first_term(y, args),Ml2_first_term(y, args),Ml3_first_term(y, args),Ml4_first_term(y, args)])
    innersums = [jnp.sum(beta_0_sti*(Is[ll]+Ia[ll])/N[ll]) for ll in range(4)]
    innersums_array = jnp.array(innersums)
    sum = jnp.sum((jnp.add(Ms,M_second_terms)*(innersums_array)))
    JP = Lambda * c[1] * sum
    return JP

def J3(y, args):
    beta_0_sti = jnp.array(args["beta_STI"])
    c = args["c"]
    M_second_terms = jnp.array([0,0,1-omega,0])
    Is = jnp.array([y["Is1_STI"],y["Is2_STI"],y["Is3_STI"],y["Is4_STI"]])
    Ia = jnp.array([y["Ia1_STI"],y["Ia2_STI"],y["Ia3_STI"],y["Ia4_STI"]])
    N = calculate_N(y)
    Ms = jnp.array([Ml1_first_term(y, args),Ml2_first_term(y, args),Ml3_first_term(y, args),Ml4_first_term(y, args)])
    innersums = [jnp.sum(beta_0_sti*(Is[ll]+Ia[ll])/N[ll]) for ll in range(4)]
    innersums_array = jnp.array(innersums)
    sum = jnp.sum((jnp.add(Ms,M_second_terms)*(innersums_array)))
    JP = Lambda * c[2] * sum
    return JP

def J4(y, args):
    beta_0_sti = jnp.array(args["beta_STI"])
    c = args["c"]
    M_second_terms = jnp.array([0,0,0,1-omega])
    Is = jnp.array([y["Is1_STI"],y["Is2_STI"],y["Is3_STI"],y["Is4_STI"]])
    Ia = jnp.array([y["Ia1_STI"],y["Ia2_STI"],y["Ia3_STI"],y["Ia4_STI"]])
    N = calculate_N(y)
    Ms = jnp.array([Ml1_first_term(y, args),Ml2_first_term(y, args),Ml3_first_term(y, args),Ml4_first_term(y, args)])
    innersums = [jnp.sum(beta_0_sti*(Is[ll]+Ia[ll])/N[ll]) for ll in range(4)]
    innersums_array = jnp.array(innersums)
    sum = jnp.sum((jnp.add(Ms,M_second_terms)*(innersums_array)))
    JP = Lambda * c[3] * sum
    return JP

# Function to calculate total number of people on PrEP
def calculate_N_Prep(y):
    return jnp.sum(jnp.array([y["SP1"],y["SP2"],y["SP3"],y["SP4"],y["IP11"],y["IP21"],y["IP31"],y["IP41"]]))
    
# Function to calculate the testing rate of STI
def lambda_a(y,args):
    contact = jnp.array(args["contact"])
    compartments = ["A11","A12","A13","A14","A21","A22","A23","A24","A31","A32","A33","A34","A41","A42","A43","A44"]
    compartment_values = jnp.array([y[key] for key in compartments])
    hazard = jnp.sum(compartment_values) / jnp.sum(calculate_N(y))
    return (
        args["lambda_0"]  # Baseline test rate
        + contact
        * (1 - m(args, y))
        * args["beta_HIV"]
        * hazard # (hazard is only last compartment of H)
        * (1 - calculate_N_Prep(y))  # HIV dependent term
        + args["lambda_P"]
        * calculate_N_Prep(y)  # Proportional infection rate due to HIV prevalence
    )


# Function to calculate infection from asymptomatic STI individuals
def infect_ia1(y, args):
    logger.debug("Calculating infection from asymptomatic STI individuals")
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J1(y, args)
    )

# Function to calculate infection from symptomatic STI individuals
def infect_is1(y, args):
    logger.debug("Calculating infection from symptomatic STI individuals")
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J1(y, args)
    )

def infect_ia2(y, args):
    logger.debug("Calculating infection from asymptomatic STI individuals")
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J2(y, args)
    )

def infect_is2(y, args):
    logger.debug("Calculating infection from symptomatic STI individuals")
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J2(y, args)
    )

def infect_ia3(y, args):
    logger.debug("Calculating infection from asymptomatic STI individuals")
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J3(y, args)
    )

def infect_is3(y, args):
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J3(y, args)
    )

def infect_ia4(y, args):
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J4(y, args)
    )

def infect_is4(y, args):
    asymptomatic = jnp.array(args["asymptomatic"])
    return (
        asymptomatic
        * (1 - m(args, y) * (1 - calculate_N_Prep(y)))
        * J4(y, args)
    )

def main_model(t, y, args):
    cm = icomo.CompModel(y)  # Initialize the compartmental model
    # unpack args
    Kons, N0s, mu, Omega, Koffs, tau, rhos, tauPs, Phi, gammas, delay = args['Kons'], args['N0s'], args['mu'], args['Omega'], args['Koffs'], args['tau'], args['rhos'], args['tauPs'], args['Phi'], args['gammas'], args['delay']
    
    # HIV dynamics-------------------------------------------------------------------------------------------------------------------------------
    # HIV dynamics double checked, not the functions that are called though, just the terms
    # Flow out of S compartments
    cm.flow("S1", "I11", JP(1,y,args))
    cm.flow("S2", "I21", JP(2,y,args))
    cm.flow("S3", "I31", JP(3,y,args))
    cm.flow("S4", "I41", JP(4,y,args))
    cm.flow("S1", "SP1", Kons[0])
    cm.flow("S2", "SP2", Kons[1])
    cm.flow("S3", "SP3", Kons[2])
    cm.flow("S4", "SP4", Kons[3])
    cm.dy["S1"] = cm.dy["S1"] - mu*y["S1"] + mu*N0s[0]
    cm.dy["S2"] = cm.dy["S2"] - mu*y["S2"] + mu*N0s[1]
    cm.dy["S3"] = cm.dy["S3"] - mu*y["S3"] + mu*N0s[2]
    cm.dy["S4"] = cm.dy["S4"] - mu*y["S4"] + mu*N0s[3]
    # Flow out of SP compartments
    cm.flow("SP1", "IP11", Omega*JP(1,y,args))
    cm.flow("SP2", "IP21", Omega*JP(2,y,args))
    cm.flow("SP3", "IP31", Omega*JP(3,y,args))
    cm.flow("SP4", "IP41", Omega*JP(4,y,args))
    cm.flow("SP1", "S1", Koffs[0])
    cm.flow("SP2", "S2", Koffs[1])
    cm.flow("SP3", "S3", Koffs[2])
    cm.flow("SP4", "S4", Koffs[3])
    cm.dy["SP1"] = cm.dy["SP1"] - mu*y["SP1"]
    cm.dy["SP2"] = cm.dy["SP2"] - mu*y["SP2"]
    cm.dy["SP3"] = cm.dy["SP3"] - mu*y["SP3"]
    cm.dy["SP4"] = cm.dy["SP4"] - mu*y["SP4"]
    # Flow out of Il1 compartments
    cm.flow("I11", "A11", tau)
    cm.flow("I21", "A21", tau)
    cm.flow("I31", "A31", tau)
    cm.flow("I41", "A41", tau)
    cm.dy["I11"] = cm.dy["I11"] - (mu + rhos[0])*y["I11"]
    cm.dy["I21"] = cm.dy["I21"] - (mu + rhos[0])*y["I21"]
    cm.dy["I31"] = cm.dy["I31"] - (mu + rhos[0])*y["I31"]
    cm.dy["I41"] = cm.dy["I41"] - (mu + rhos[0])*y["I41"]
    # Flow out of IPl1 compartments
    cm.flow("IP11", "A11", tauPs[0])
    cm.flow("IP21", "A21", tauPs[1])
    cm.flow("IP31", "A31", tauPs[2])
    cm.flow("IP41", "A41", tauPs[3])
    cm.dy["IP11"] = cm.dy["IP11"] - mu*y["IP11"]
    cm.dy["IP21"] = cm.dy["IP21"] - mu*y["IP21"]
    cm.dy["IP31"] = cm.dy["IP31"] - mu*y["IP31"]
    cm.dy["IP41"] = cm.dy["IP41"] - mu*y["IP41"]
    # Flow out of Ilk compartments (k=2,3,4)
    # k=2
    cm.flow("I12", "A12", tau)
    cm.flow("I22", "A22", tau)
    cm.flow("I32", "A32", tau) 
    cm.flow("I42", "A42", tau)
    cm.dy["I12"] = cm.dy["I12"] - (mu + rhos[1])*y["I12"] + rhos[0]*y["I11"]
    cm.dy["I22"] = cm.dy["I22"] - (mu + rhos[1])*y["I22"] + rhos[0]*y["I21"]
    cm.dy["I32"] = cm.dy["I32"] - (mu + rhos[1])*y["I32"] + rhos[0]*y["I31"]
    cm.dy["I42"] = cm.dy["I42"] - (mu + rhos[1])*y["I42"] + rhos[0]*y["I41"]
    # k=3
    cm.flow("I13", "A13", tau)
    cm.flow("I23", "A23", tau)
    cm.flow("I33", "A33", tau)
    cm.flow("I43", "A43", tau)
    cm.dy["I13"] = cm.dy["I13"] - (mu + rhos[2])*y["I13"] + rhos[1]*y["I12"]
    cm.dy["I23"] = cm.dy["I23"] - (mu + rhos[2])*y["I23"] + rhos[1]*y["I22"]
    cm.dy["I33"] = cm.dy["I33"] - (mu + rhos[2])*y["I33"] + rhos[1]*y["I32"]
    cm.dy["I43"] = cm.dy["I43"] - (mu + rhos[2])*y["I43"] + rhos[1]*y["I42"]
    # k=4
    cm.flow("I14", "A14", tau)
    cm.flow("I24", "A24", tau)
    cm.flow("I34", "A34", tau)
    cm.flow("I44", "A44", tau)
    cm.dy["I14"] = cm.dy["I14"] - (mu + rhos[3])*y["I14"] + rhos[2]*y["I13"]
    cm.dy["I24"] = cm.dy["I24"] - (mu + rhos[3])*y["I24"] + rhos[2]*y["I23"]
    cm.dy["I34"] = cm.dy["I34"] - (mu + rhos[3])*y["I34"] + rhos[2]*y["I33"]
    cm.dy["I44"] = cm.dy["I44"] - (mu + rhos[3])*y["I44"] + rhos[2]*y["I43"]
    # Flow out of Al1 compartments
    cm.flow("A11", "I11", Phi)
    cm.flow("A21", "I21", Phi)
    cm.flow("A31", "I31", Phi)
    cm.flow("A41", "I41", Phi)
    cm.dy["A11"] = cm.dy["A11"] - (mu + gammas[0])*y["A11"]
    cm.dy["A21"] = cm.dy["A21"] - (mu + gammas[0])*y["A21"]
    cm.dy["A31"] = cm.dy["A31"] - (mu + gammas[0])*y["A31"]
    cm.dy["A41"] = cm.dy["A41"] - (mu + gammas[0])*y["A41"]
    # Flow out of Alk compartments (k=2,3,4)
    # k=2
    cm.flow("A12", "I12", Phi)
    cm.flow("A22", "I22", Phi)
    cm.flow("A32", "I32", Phi)
    cm.flow("A42", "I42", Phi)
    cm.dy["A12"] = cm.dy["A12"] - (mu + gammas[1])*y["A12"] + gammas[0]*y["A11"]
    cm.dy["A22"] = cm.dy["A22"] - (mu + gammas[1])*y["A22"] + gammas[0]*y["A21"]
    cm.dy["A32"] = cm.dy["A32"] - (mu + gammas[1])*y["A32"] + gammas[0]*y["A31"]
    cm.dy["A42"] = cm.dy["A42"] - (mu + gammas[1])*y["A42"] + gammas[0]*y["A41"]
    # k=3
    cm.flow("A13", "I13", Phi)
    cm.flow("A23", "I23", Phi)
    cm.flow("A33", "I33", Phi)
    cm.flow("A43", "I43", Phi)
    cm.dy["A13"] = cm.dy["A13"] - (mu + gammas[2])*y["A13"] + gammas[1]*y["A12"]
    cm.dy["A23"] = cm.dy["A23"] - (mu + gammas[2])*y["A23"] + gammas[1]*y["A22"]
    cm.dy["A33"] = cm.dy["A33"] - (mu + gammas[2])*y["A33"] + gammas[1]*y["A32"]
    cm.dy["A43"] = cm.dy["A43"] - (mu + gammas[2])*y["A43"] + gammas[1]*y["A42"]
    # k=4
    cm.flow("A14", "I14", Phi)
    cm.flow("A24", "I24", Phi)
    cm.flow("A34", "I34", Phi)
    cm.flow("A44", "I44", Phi)
    cm.dy["A14"] = cm.dy["A14"] - (mu + gammas[3])*y["A14"] + gammas[2]*y["A13"]
    cm.dy["A24"] = cm.dy["A24"] - (mu + gammas[3])*y["A24"] + gammas[2]*y["A23"]
    cm.dy["A34"] = cm.dy["A34"] - (mu + gammas[3])*y["A34"] + gammas[2]*y["A33"]
    cm.dy["A44"] = cm.dy["A44"] - (mu + gammas[3])*y["A44"] + gammas[2]*y["A43"]

    # hazard--------------------------------------------------------------------------------------------------------------------------------------------------------
    #compartments = ["A11","A12","A13","A14","A21","A22","A23","A24","A31","A32","A33","A34","A41","A42","A43","A44"]
    #tau_delay = jnp.array([tau for _ in compartments]) # Create an array of delays with the same length as compartments

    #for start_comp in compartments:
        # # y[start_comp] = jnp.array(y[start_comp])
        # if not isinstance(y[start_comp], jnp.ndarray):
        #     raise ValueError(f"{start_comp} must be an array-like object")
        #cm.delayed_copy(start_comp,"H", delay)
    # temp = ["A11","A12","A13","A14","A21","A22","A23","A24","A31","A32","A33","A34","A41","A42","A43","A44"]
    # cm.delayed_copy(temp,"H", jnp.array([tau, tau]))

    # STI dynamics-------------------------------------------------------------------------------------------------------------------------------------------------
    # Basic STI dynamics
    cm.flow("S1_STI", "Ia1_STI", infect_ia1(y, args)) # Susceptible to asymptomatic
    cm.flow("S1_STI", "Is1_STI", infect_is1(y, args)) # Susceptible to symptomatic
    cm.flow("Ia1_STI", "S1_STI", args["gamma_STI"]) # Asymptomatic to susceptible (recovery)
    cm.flow("Ia1_STI", "T1_STI", lambda_a(y,args)) # Asymptomatic to tested and treatment
    cm.flow("Is1_STI", "T1_STI", args["lambda_s"]) # Symptomatic to tested and treatment
    cm.flow("T1_STI", "S1_STI", args["gammaT_STI"]) # Treatment to susceptible (immunity loss)
    # Vital dynamics (New addition/removoval to/from risk group)
    cm.flow("S2_STI", "Ia2_STI", infect_ia2(y, args)) # Susceptible to asymptomatic
    cm.flow("S2_STI", "Is2_STI", infect_is2(y, args)) # Susceptible to symptomatic
    cm.flow("Ia2_STI", "S2_STI", args["gamma_STI"]) # Asymptomatic to susceptible (recovery)
    cm.flow("Ia2_STI", "T2_STI", lambda_a(y,args)) # Asymptomatic to tested and treatment
    cm.flow("Is2_STI", "T2_STI", args["lambda_s"]) # Symptomatic to tested and treatment
    cm.flow("T2_STI", "S2_STI", args["gammaT_STI"]) # Treatment to susceptible (immunity loss)
    cm.dy["S1_STI"] = cm.dy["S1_STI"] - mu * y["S1_STI"] + mu * N0s[0]
    cm.dy["Ia1_STI"] = cm.dy["Ia1_STI"] - mu * y["Ia1_STI"]
    cm.dy["Is1_STI"] = cm.dy["Is1_STI"] - mu * y["Is1_STI"] 
    cm.dy["T1_STI"] = cm.dy["T1_STI"] - mu * y["T1_STI"]
    # Vital dynamics (New addition/removal to/from risk group)
    cm.flow("S3_STI", "Ia3_STI", infect_ia3(y, args)) # Susceptible to asymptomatic
    cm.flow("S3_STI", "Is3_STI", infect_is3(y, args)) # Susceptible to symptomatic
    cm.flow("Ia3_STI", "S3_STI", args["gamma_STI"]) # Asymptomatic to susceptible (recovery)
    cm.flow("Ia3_STI", "T3_STI", lambda_a(y,args)) # Asymptomatic to tested and treatment
    cm.flow("Is3_STI", "T3_STI", args["lambda_s"]) # Symptomatic to tested and treatment
    cm.flow("T3_STI", "S3_STI", args["gammaT_STI"]) # Treatment to susceptible (immunity loss)
    cm.dy["S2_STI"] = cm.dy["S2_STI"] - mu * y["S2_STI"] + mu * N0s[1]
    cm.dy["Ia2_STI"] = cm.dy["Ia2_STI"] - mu * y["Ia2_STI"]
    cm.dy["Is2_STI"] = cm.dy["Is2_STI"] - mu * y["Is2_STI"]
    cm.dy["T2_STI"] = cm.dy["T2_STI"] - mu * y["T2_STI"]
    # Vital dynamics (New addition/removal to/from risk group)
    cm.flow("S4_STI", "Ia4_STI", infect_ia4(y, args)) # Susceptible to asymptomatic
    cm.flow("S4_STI", "Is4_STI", infect_is4(y, args)) # Susceptible to symptomatic
    cm.flow("Ia4_STI", "S4_STI", args["gamma_STI"]) # Asymptomatic to susceptible (recovery)
    cm.flow("Ia4_STI", "T4_STI", lambda_a(y,args)) # Asymptomatic to tested and treatment
    cm.flow("Is4_STI", "T4_STI", args["lambda_s"]) # Symptomatic to tested and treatment
    cm.flow("T4_STI", "S4_STI", args["gammaT_STI"]) # Treatment to susceptible (immunity loss)
    cm.dy["S3_STI"] = cm.dy["S3_STI"] - mu * y["S3_STI"] + mu * N0s[2]
    cm.dy["Ia3_STI"] = cm.dy["Ia3_STI"] - mu * y["Ia3_STI"]
    cm.dy["Is3_STI"] = cm.dy["Is3_STI"] - mu * y["Is3_STI"]
    cm.dy["T3_STI"] = cm.dy["T3_STI"] - mu * y["T3_STI"]
    # Vital dynamics (New addition/removal to/from risk group)
    cm.dy["S4_STI"] = cm.dy["S4_STI"] - mu * y["S4_STI"] + mu * N0s[3]
    cm.dy["Ia4_STI"] = cm.dy["Ia4_STI"] - mu * y["Ia4_STI"]
    cm.dy["Is4_STI"] = cm.dy["Is4_STI"] - mu * y["Is4_STI"]
    cm.dy["T4_STI"] = cm.dy["T4_STI"] - mu * y["T4_STI"]

    # Return the differential changes
    return cm.dy

# Function to setup the model and return the integrator
def setup_model():
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
        t_0=ts[0],  # Initial time point
        t_1=ts[-1],  # Final time point
        ts_solver=ts,  # Time points for the solver to use
    )

    # Get the integration function for the model
    integrator = integrator_object.get_func(main_model)  # Returns a function that can be used to solve the ODEs defined in the 'model' function

    logger.info("Model setup complete and ready for simulation")

    return integrator