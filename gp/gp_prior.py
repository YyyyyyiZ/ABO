import numpy as np


def get_prior(prior_func, *args):
    """
    Get a prior function with extra arguments bound.

    Parameters:
    prior_func (function): The prior function to bind arguments to.
    *args: The arguments to bind to the prior function.

    Returns:
    function: A function that calls the original prior function with the bound arguments.
    """

    def prior(*varargs):
        return prior_func(*args, *varargs)

    return prior


def gaussian_prior(mean=None, variance=None, theta=None):
    """
    Gaussian prior distribution.

    Parameters:
    mean (float): Mean parameter of the Gaussian distribution.
    variance (float): Variance parameter of the Gaussian distribution.
    theta (array-like): Query hyperparameters for prior evaluation.

    Returns:
    If theta is None, return a random sample from the Gaussian distribution.
    If theta is provided, return the log-likelihood (result) and its derivative (dlp).
    """
    if mean is None or variance is None:
        raise ValueError('mean and variance parameters need to be provided')

    if theta is None:
        # Return a random sample
        result = priorGauss(mean, variance)
        return result

    # Compute log-likelihood and its derivative
    result, dlp = priorGauss(mean, variance, theta)

    # If second derivative is not requested, we are done
    if len(dlp) == 1:
        return result

    # Second derivative of log likelihood
    d2lp = -np.ones_like(result) / variance

    return result, dlp, d2lp


def priorGauss(mu, s2, x=None):
    """
    Univariate Gaussian hyperparameter prior distribution.
    Compute log-likelihood and its derivative or draw a random sample.

    Parameters:
    mu (float): Mean parameter of the Gaussian distribution.
    s2 (float): Variance parameter of the Gaussian distribution.
    x (array-like, optional): Query hyperparameters for prior evaluation.

    Returns:
    If x is None, return a random sample from the Gaussian distribution.
    If x is provided, return the log-likelihood (lp) and its derivative (dlp).
    """
    if mu is None or s2 is None:
        raise ValueError('mu and s2 parameters need to be provided')

    if not (np.isscalar(mu) and np.isscalar(s2)):
        raise ValueError('mu and s2 parameters need to be scalars')

    if x is None:
        # Return a random sample
        return np.sqrt(s2) * np.random.randn() + mu

    # Compute log-likelihood and its derivative
    lp = -((x - mu) ** 2 / (2 * s2)) - np.log(2 * np.pi * s2) / 2
    dlp = -(x - mu) / s2

    return lp, dlp
