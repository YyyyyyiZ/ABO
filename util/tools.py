import random
import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm


def sobol_sample(n=100, d=1):
    sampler = Sobol(d, scramble=True)
    samples = sampler.random(n)
    return samples


def prior_sample(priors, samples):
    n = samples.shape[0]
    prior_mean = np.zeros(len(priors))
    prior_std = np.zeros(len(priors))

    for i, prior in enumerate(priors):
        prior_func = prior['function']
        prior_args = prior['args']
        prior_mean[i] = prior_args[0]
        prior_std[i] = np.sqrt(prior_args[1])

    hyps = norm.ppf(samples[:, :len(priors)], loc=np.tile(prior_mean, (n, 1)), scale=np.tile(prior_std, (n, 1)))
    return hyps


def fix_pd_matrix(A, epsilon=1E-6):
    n, m = A.shape
    # Check if A is a square matrix
    if n != m:
        raise ValueError("This matrix is not square")
    # Ensure the matrix is symmetric
    A = (A + A.T) / 2
    # Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    # Fix the eigenvalues
    eigvals[eigvals < epsilon] = epsilon
    A_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Ensure the result is still symmetric
    A_fixed = (A_fixed + A_fixed.T) / 2

    return A_fixed


def hellinger_distance(p_mu, p_K, q_mu, q_K):
    # Squared Hellinger distance for two multivariate Gaussian distributions
    # p_K and q_K are the same, just return 0
    if np.allclose(p_K, q_K, atol=np.finfo(float).eps):
        return 0
    try:
        chol_p = np.linalg.cholesky(p_K)
        chol_q = np.linalg.cholesky(q_K)
    except np.linalg.LinAlgError:
        try:
            chol_p = np.linalg.cholesky(fix_pd_matrix(p_K))
            chol_q = np.linalg.cholesky(fix_pd_matrix(q_K))
        except np.linalg.LinAlgError:
            chol_p = fix_pd_matrix(smoothing_matrix(p_K))
            chol_q = fix_pd_matrix(smoothing_matrix(q_K))

    # Compute Hellinger distance using logdet
    # Compute P logdet
    logdet_p = 2 * np.sum(np.log(np.diag(chol_p)))
    # Compute Q logdet
    logdet_q = 2 * np.sum(np.log(np.diag(chol_q)))
    # Compute .5(P+Q) logdet
    chol_pq = np.linalg.cholesky(0.5 * (p_K + q_K))
    logdet_pq = 2 * np.sum(np.log(np.diag(chol_pq)))
    # Compute log distance
    log_base = 0.25 * (logdet_p + logdet_q) - 0.5 * logdet_pq
    mu = p_mu - q_mu

    if np.all(mu < np.finfo(float).eps):
        log_h = log_base
    else:
        log_h = log_base - (1/8) * mu.T @ np.linalg.solve(chol_pq.T, np.linalg.solve(chol_pq, mu))

    # Exponentiate
    h = 1 - np.exp(log_h)

    return h


def smoothing_matrix(A):
    # First we compute the squared Frobenius norm of our matrix
    nA = np.sum(A**2)
    # Then we make this norm be meaningful for element wise comparison
    nA = nA / A.size
    # Finally, we smooth our matrix
    As = A.copy()
    As[As**2 < 1e-10 * nA] = 0
    return As

def get_num_hyps(names, base_names, base_hyps):
    """
    Calculate the number of hyperparameters for each name based on the base names and their corresponding hyperparameters.

    Parameters:
    names (list of str): List of names to evaluate.
    base_names (list of str): List of base names.
    base_hyps (list of int): List of hyperparameters corresponding to each base name.

    Returns:
    list of int: List of number of hyperparameters for each name.
    """
    num_hyps = [0] * len(names)
    for i, name in enumerate(names):
        count = 0
        for k, base_name in enumerate(base_names):
            count += base_hyps[k] * name.count(base_name)
        num_hyps[i] = count
    return num_hyps




