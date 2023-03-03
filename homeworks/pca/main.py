from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a reconstruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    raise NotImplementedError("Your Code Goes Here")


# @problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    This function has been implemented for you.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    tmp = demean_data - reconstruct_demean(uk, demean_data)
    # take the norm of each column, and then find the mean (of n values)
    res = np.mean(np.linalg.norm(tmp, axis=1) ** 2)
    return res


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of its covariance matrix.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,). Should be in descending order.
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    cov_mat = demean_data.T @ demean_data / len(demean_data)
    return np.linalg.eig(cov_mat)


@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.

    Note that you do not need to use reconstruction_error anywhere in the Winter 2023 iteration of this course.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")

    raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
