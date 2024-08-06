import numpy as np
import logging
import jax
from tapm import utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_sti_infections(
    Hs, Ps, lambda_P_values, y0, args, integrator, model_STI, filename=None
):
    """
    Compute STI infections for given H, P, and lambda_P values and store the results for different H values.

    Parameters:
    Hs (list): List of H values.
    Ps (list): List of P values.
    lambda_P_values (list): List of lambda_P values.
    y0 (dict): Initial state of the system.
    args (dict): Arguments for the model.
    integrator (function): Function to integrate the model.
    model_STI (module): Module containing STI model functions.
    filename (str): Name of the file to save the results.

    Returns:
    dict: Results for different H values.
    """
    # Dictionary to store results for different H values
    results = {}

    # Loop over each value of H
    for H in Hs:
        logger.info(f"Processing H={H}")

        # Determine the size of the result matrices
        res_size = [len(lambda_P_values), len(Ps)]

        # Initialize result matrices
        res_Ia = np.zeros(res_size)  # Asymptomatic STI infections
        res_Is = np.zeros(res_size)  # Symptomatic STI infections
        res_T = np.zeros(res_size)  # Treated STI infections
        res_infections = np.zeros(res_size)  # Total infections
        res_asymp_infections = np.zeros(res_size)  # Asymptomatic infections
        res_symp_infections = np.zeros(res_size)  # Symptomatic infections
        res_tests = np.zeros(res_size)  # Number of tests
        res_asymp_tests = np.zeros(res_size)  # Tests for asymptomatic
        res_symp_tests = np.zeros(res_size)  # Tests for symptomatic
        check = np.zeros(res_size)  # Convergence check

        # Loop over each lambda_P value
        for i, lambda_P in enumerate(lambda_P_values):
            for j, P in enumerate(Ps):
                args_mod = args.copy()
                args_mod["H"] = H
                args_mod["P_HIV"] = P
                args_mod["lambda_P"] = lambda_P

                # Integrate the model with the modified arguments
                output = integrator(y0=y0, constant_args=args_mod)

                # Get the final state of the system
                y1 = {key: value[-1] for key, value in output.items()}

                # Record the final asymptomatic STI infections
                res_Ia[i, j] = output["Ia_STI"][-1]

                # Record the final symptomatic STI infections
                res_Is[i, j] = output["Is_STI"][-1]

                # Record the final treated STI infections
                res_T[i, j] = output["T_STI"][-1]

                # Calculate total new infections
                res_infections[i, j] = (
                    model_STI.infect_is(y1, args_mod)
                    + model_STI.infect_ia(y1, args_mod)
                ) * y1["S_STI"]

                # Calculate new asymptomatic infections
                res_asymp_infections[i, j] = (
                    model_STI.infect_ia(y1, args_mod) * y1["S_STI"]
                )

                # Calculate new symptomatic infections
                res_symp_infections[i, j] = (
                    model_STI.infect_is(y1, args_mod) * y1["S_STI"]
                )

                # Detected (by testing) new infections from symptomatic and asymptomatic
                res_tests[i, j] = (model_STI.lambda_a(args_mod) * y1["Ia_STI"]) + (
                    args["lambda_s"] * y1["Is_STI"]
                )

                # Detected (by testing) new infections from asymptomatic
                res_asymp_tests[i, j] = model_STI.lambda_a(args_mod) * y1["Ia_STI"]

                # Detected (by testing) new infections from symptomatic
                res_symp_tests[i, j] = args["lambda_s"] * y1["Is_STI"]

                # Check for convergence by comparing the last and the second last values
                check[i, j] = (
                    abs(output["Ia_STI"][-1] - output["Ia_STI"][-len(lambda_P_values)])
                    + abs(output["T_STI"][-1] - output["T_STI"][-len(lambda_P_values)])
                    + abs(output["Is_STI"][-1] - output["Is_STI"][-len(lambda_P_values)])
                )

        # Store the results for the current H value
        results[H] = {
            "res_Ia": res_Ia,
            "res_Is": res_Is,
            "res_T": res_T,
            "res_infections": res_infections,
            "res_asymp_infections": res_asymp_infections,
            "res_symp_infections": res_symp_infections,
            "res_tests": res_tests,
            "res_asymp_tests": res_asymp_tests,
            "res_symp_tests": res_symp_tests,
            "check": check,
        }

        # Log the maximum value from the check matrix to ensure convergence
        logger.info(f"Max check value for H={H}: {results[H]['check'].max()}")

    # Save the results to a file
    if filename:
        utils.save_results(results, filename)

    return results
