import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import jax

import icomo


def calculate_m(args, H):
    return args["min_exp"] + (args["max_exp"] - args["min_exp"]) * (
        1 - np.exp(-H / args["tau_exp"])
    )


def calculate_lambda_a(args, lambda_P, m, P, H):
    return (lambda_P * P) + (args["c"] * (1 - m) * args["beta_HIV"] * H * (1 - P))


# Alpha
def calculate_alpha(args, P, m):
    return args["beta_STI"] * (1 - m * (1 - P))


def calculate_a(args):
    return args["lambda_s"] + args["mu"]


def calculate_b(args, lambda_a):
    return args["gamma_STI"] + lambda_a + args["mu"]


def calculate_I_Assymp(args, lambda_P, H, P):
    # Calculate a, b, and R_0
    m = calculate_m(args, H)
    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    a = calculate_a(args)
    b = calculate_b(args, lambda_a)
    varphi = b + a * args["asymptomatic"] - b * args["asymptomatic"]
    Ro = calculate_Ro(args, lambda_P, H, P)
    # Calculate I_STI^a,*
    numerator = (
        (args["mu"] + args["gammaT_STI"]) * a * args["asymptomatic"] * (1 - 1 / Ro)
    )
    denominator = (
        (a * b)
        - (a * args["gamma_STI"] * args["asymptomatic"])
        + (args["gammaT_STI"] * varphi)
    )

    I_STI_a_star = numerator / denominator

    return I_STI_a_star


def calculate_S(args, lambda_P, H, P):
    m = calculate_m(args, H)
    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    alpha = calculate_alpha(args, P, m)
    a = calculate_a(args)
    b = calculate_b(args, lambda_a)
    varphi = b + a * args["asymptomatic"] - b * args["asymptomatic"]

    S_math = a * b / (alpha * (varphi))

    return S_math


def calculate_Ro(args, lambda_P, H, P):
    S = calculate_S(args, lambda_P, H, P)

    Ro = 1 / S

    return Ro


def calculate_I_Symp(args, lambda_P, H, P):
    m = calculate_m(args, H)
    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    a = calculate_a(args)
    b = calculate_b(args, lambda_a)
    asym = calculate_I_Assymp(args, lambda_P, H, P)
    return asym * (b / a) * (1 - args["asymptomatic"]) / args["asymptomatic"]


def calculate_y0(args, lambda_P, H, P):
    S = calculate_S(args, lambda_P, H, P)
    I_s = calculate_I_Symp(args, lambda_P, H, P)
    I_a = calculate_I_Assymp(args, lambda_P, H, P)
    T = 1 - S - I_s - I_a
    y0 = {
        "S_STI": S - 0.1,  # Initial susceptible proportion
        "Ia_STI": I_a + 0.09,  # Initial asymptomatic proportion
        "Is_STI": I_s + 0.01,  # Initial symptomatic proportion
        "T_STI": T,  # Initial treated proportion
    }
    return y0
