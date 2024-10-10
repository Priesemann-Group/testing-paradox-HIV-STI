import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tapm import utils

# Parameters





def calculate_m(H, args):
    min_exp = args["min_exp"]
    max_exp = args["max_exp"]
    tau_exp = args["tau_exp"]
    return min_exp + (max_exp - min_exp) * (1 - np.exp(-H / tau_exp))

def calculate_lambda_a(args, lambda_P, m, P, H):
    return (lambda_P * P) + (args["c"] * (1 - m) * args["beta_HIV"] * H * (1 - P))

def calculate_lambda_s(args, lambda_a):
    return args["lambda_s"] + lambda_a

# Alpha
def calculate_alpha(P, m, args):
    return args["beta_STI"] * (1 - m * (1 - P))


def calculate_a(args, lambda_s):
    return lambda_s + args["mu"]


def calculate_b(args, lambda_a):
    return args["gamma_STI"] + lambda_a + args["mu"]


def calculate_R_0(args, b, alpha, a):
    return (
        ((args["asymptomatic"] * alpha * a / b) + (1 - args["asymptomatic"]) * alpha)
    ) / a


def calculate_Omega(args, b, a):
    return (b / a) * (1 - args["asymptomatic"]) / args["asymptomatic"]


def calculate_I_STI_a(args, lambda_P, H, P):

    # Calculate a, b, and R_0

    m = calculate_m(H,args)

    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    lambda_s = calculate_lambda_s(args, lambda_a)

    alpha = calculate_alpha(P, m, args)
    a = calculate_a(args, lambda_s)
    b = calculate_b(args, lambda_a)

    R_0 = calculate_R_0(args, b, alpha, a)

    # Calculate I_STI^a,*

    numerator = (args["mu"] + args["gammaT_STI"]) * (1 / R_0 - 1)

    denominator = (
        -b / args["asymptomatic"]
        + args["gamma_STI"]
        - args["gammaT_STI"]
        * (1 + (b / a) * (1 - args["asymptomatic"]) / args["asymptomatic"])
    )

    I_STI_a_star = numerator / denominator

    return I_STI_a_star


def calculate_I_Assymp(args, lambda_P, H, P):

    # Calculate a, b, and R_0

    m = calculate_m(H,args)

    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    lambda_s = calculate_lambda_s(args, lambda_a)

    a = calculate_a(args, lambda_s)
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

    m = calculate_m(H,args)

    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    lambda_s = calculate_lambda_s(args, lambda_a)

    alpha = calculate_alpha(P, m, args)
    a = calculate_a(args, lambda_s)
    b = calculate_b(args, lambda_a)

    varphi = b + a * args["asymptomatic"] - b * args["asymptomatic"]

    S_math = a * b / (alpha * (varphi))

    return S_math


def calculate_Ro(args, lambda_P, H, P):
    S = calculate_S(args, lambda_P, H, P)

    Ro = 1 / S

    return Ro


def calculate_I_Symp(args, lambda_P, H, P):

    m = calculate_m(H,args)

    lambda_a = calculate_lambda_a(args, lambda_P, m, P, H)
    lambda_s = calculate_lambda_s(args, lambda_a)
    a = calculate_a(args, lambda_s)
    b = calculate_b(args, lambda_a)

    asym = calculate_I_Assymp(args, lambda_P, H, P)

    return asym * (b / a) * (1 - args["asymptomatic"]) / args["asymptomatic"]