import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Save the results for different H values
def save_results(results, filename):
    """
    Save the results to a file.

    Parameters:
    results (dict): Results to be saved.
    filename (str): Name of the file to save the results.
    """
    # Save the results to a file
    np.save(f"../results/{filename}.npy", results)
    logger.info(f"Results saved to ../results/{filename}.npy")

# Load the results from a file
def load_results(filename):
    """
    Load the results from a file.

    Parameters:
    filename (str): Name of the file to load the results from.

    Returns:
    dict: Loaded results.
    """
    # Load the results from a file
    results = np.load(f"../results/{filename}.npy", allow_pickle=True).item()
    logger.info(f"Results loaded from ../results/{filename}.npy")
    return results


# Define a function to read and evaluate the file
# PD: using np and jnp in calculation of args in HIVandSTI.py (in fraction2rate() etc.)
#     this breaks this usage of exec with empty namespace
def read_params(filename):
    with open(filename, 'r') as file:
        content = file.read()
    # Create a dictionary to hold the local variables
    local_vars = {}
    # Execute the content of the file
    exec(content, {}, local_vars)
    # Extract the variables
    return local_vars['args'], local_vars['y0']