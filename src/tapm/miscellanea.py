import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tapm import utils

# Parameters





def kappa(P, H, args):
    return args['beta_STI'] * (1 - m(P, H, args) * (1 - P))

def lambdas(P, H, args, lambda_P):
    return args["lambda_s"] + lambdaa(P, H, args, lambda_P)

def lambdaa(P, H, args, lambda_P):
    return lambdaH(P, H, args) * (1 - P) + lambda_P * P

def lambdaH(P, H, args):
    return args["c"] * (1 - m(P, H, args)) * args["beta_HIV"] * H

def m(P, H, args):
    return args["min_exp"] + (args["max_exp"] - args["min_exp"]) * (1 - np.exp(-H / args["tau_exp"]))

def calculate_Ia(P, H, args, lambda_P):
    gamma, tilde_gamma, mu, psi, Sigma = args["gamma_STI"], args["gammaT_STI"], args["mu"], args["psi"], args["Sigma"]
    kappa_val = kappa(P, H, args)
    lambda_s_val = lambdas(P, H, args, lambda_P)
    lambda_a_val = lambdaa(P, H, args, lambda_P)

    C = (gamma + lambda_a_val + mu) * (1 - psi) / (psi * (lambda_s_val + mu))
    A = kappa_val * (1 + C) * (-(lambda_s_val + mu) * C + (gamma - tilde_gamma * (1 + C)) * (1 - psi))
    B = (tilde_gamma + mu) * ((1 - psi) * kappa_val * (1 + C) - (lambda_s_val + mu) * C)
    D = (tilde_gamma + mu) * (1 - psi) * Sigma

    discriminant = B**2 - 4 * A * D
    if discriminant < 0:
        raise ValueError("Discriminant is negative, no real roots exist.")

    Ia_star = (-B - np.sqrt(discriminant)) / (2 * A)

    return Ia_star

def calculate_Is(P, H, args, lambda_P):
    gamma, tilde_gamma, mu, psi, Sigma = args["gamma_STI"], args["gammaT_STI"], args["mu"], args["psi"], args["Sigma"]
    kappa_val = kappa(P, H, args)
    lambda_s_val = lambdas(P, H, args, lambda_P)
    lambda_a_val = lambdaa(P, H, args, lambda_P)

    C = (gamma + lambda_a_val + mu) * (1 - psi) / (psi * (lambda_s_val + mu))
    A = kappa_val * (1 + C) * (-(lambda_s_val + mu) * C + (gamma - tilde_gamma * (1 + C)) * (1 - psi))
    B = (tilde_gamma + mu) * ((1 - psi) * kappa_val * (1 + C) - (lambda_s_val + mu) * C)
    D = (tilde_gamma + mu) * (1 - psi) * Sigma

    discriminant = B**2 - 4 * A * D
    if discriminant < 0:
        raise ValueError("Discriminant is negative, no real roots exist.")

    Ia_star = (-B - np.sqrt(discriminant)) / (2 * A)
    Is_star = C * Ia_star

    return Is_star

def calculate_S(P, H, args, lambda_P):
    gamma, tilde_gamma, mu, psi, Sigma = args["gamma_STI"], args["gammaT_STI"], args["mu"], args["psi"], args["Sigma"]
    kappa_val = kappa(P, H, args)
    lambda_s_val = lambdas(P, H, args, lambda_P)
    lambda_a_val = lambdaa(P, H, args, lambda_P)

    C = (gamma + lambda_a_val + mu) * (1 - psi) / (psi * (lambda_s_val + mu))
    A = kappa_val * (1 + C) * (-(lambda_s_val + mu) * C + (gamma - tilde_gamma * (1 + C)) * (1 - psi))
    B = (tilde_gamma + mu) * ((1 - psi) * kappa_val * (1 + C) - (lambda_s_val + mu) * C)
    D = (tilde_gamma + mu) * (1 - psi) * Sigma

    discriminant = B**2 - 4 * A * D
    if discriminant < 0:
        raise ValueError("Discriminant is negative, no real roots exist.")

    Ia_star = (-B - np.sqrt(discriminant)) / (2 * A)
    S_star = ((lambda_s_val + mu) * C * Ia_star - (1 - psi) * Sigma) / ((1 - psi) * kappa_val * (1 + C) * Ia_star)

    return S_star

def R0(P, H, args, lambda_P):
    return args["psi"] * (args['beta_STI'] * (1 - m(P, H, args) * (1 - P))) / (args["gamma_STI"] + lambdaa(P, H, args, lambda_P) + args["mu"]) + (1 - args["psi"]) * (args['beta_STI'] * (1 - m(P, H, args) * (1 - P))) / (lambdas(P, H, args, lambda_P) + args["mu"])
