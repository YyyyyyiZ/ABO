import GPy
import numpy as np


def factor_sqdexp_covariance(input_dim):
    return GPy.kern.RBF(input_dim)


def ard_sqdexp_covariance(input_dim):
    return GPy.kern.RBF(input_dim, ARD=True)


def isotropic_sqdexp_covariance(input_dim):
    return GPy.kern.RBF(input_dim, ARD=False)


def isotropic_matern_covariance(input_dim):
    return GPy.kern.Matern32(input_dim)


def periodic_covariance(input_dim):
    return GPy.kern.StdPeriodic(input_dim)


def linear_covariance(input_dim):
    return GPy.kern.Linear(input_dim)


def isotropic_rq_covariance(input_dim):
    return GPy.kern.RatQuad(input_dim)