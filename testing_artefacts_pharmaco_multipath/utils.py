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
