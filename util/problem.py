import numpy as np
import GPy
from gp.gp_prior import get_prior, gaussian_prior
from covariance.kernal import *


class Problem:
    def __init__(self, fun, lb, ub, d, optimum, covariances_root=None, num_queries=50,
                 max_num_models=50, prediction_function='gp', max_candidates=200, boms_eval_budget=5,
                 exploit_budget=5, explore_budget=10, total_hyp_samples=100, param_k=3, data_noise=0.01):
        if covariances_root is None:
            covariances_root = ['SE', 'RQ']

        self.function_name = fun
        self.lb = lb
        self.ub = ub
        self.d = d
        self.optimum = optimum

        self.covariance_root = covariances_root
        self.budget = num_queries
        self.max_num_models = max_num_models
        self.prediction_function = prediction_function
        self.max_candidates = max_candidates
        self.eval_budget = boms_eval_budget
        self.exploit_budget = exploit_budget
        self.explore_budget = explore_budget
        self.total_hyp_samples = total_hyp_samples
        self.k = param_k
        self.data_noise = data_noise

        self.initial_x = None
        self.initial_y = None
        self.theta_models = None
        self.n = None
        self.points = None
        self.y = None
        self.covariances = None


class Theta:
    def __init__(self, data_noise=0.01):
        self.length_scale_mean = np.log(0.1)
        self.length_scale_var = 1

        self.output_scale_mean = np.log(0.4)
        self.output_scale_var = 1

        self.p_length_scale_mean = np.log(2)
        self.p_length_scale_var = 0.5

        self.p_mean = np.log(0.1)
        self.p_var = 0.5

        self.alpha_mean = np.log(0.05)
        self.alpha_var = 0.5

        self.lik_noise_std = np.log(data_noise)
        self.lik_noise_std_var = 1

        self.mean_offset = 0
        self.mean_var = 1


class Covariance:
    def __init__(self, name=None, hyp=None, *varargin):
        if name == 'SEfactor':
            d = 2 * varargin[0]
            priors = [get_prior(gaussian_prior, hyp['length_scale_mean'], hyp['length_scale_var']) for _ in range(d)]
            self.fun = factor_sqdexp_covariance
            self.priors = priors
        elif name == 'SEard':
            d = varargin[0]
            priors = [get_prior(gaussian_prior, hyp['length_scale_mean'], hyp['length_scale_var']) for _ in range(d)]
            priors.append(get_prior(gaussian_prior, hyp['output_scale_mean'], hyp['output_scale_var']))
            self.fun = ard_sqdexp_covariance
            self.priors = priors
        elif name == 'SE':
            priors = [get_prior(gaussian_prior, hyp['length_scale_mean'], hyp['length_scale_var']),
                      get_prior(gaussian_prior, hyp['output_scale_mean'], hyp['output_scale_var'])]
            self.fun = isotropic_sqdexp_covariance
            self.priors = priors
        elif name == 'M1':
            priors = [get_prior(gaussian_prior, hyp['length_scale_mean'], hyp['length_scale_var']),
                      get_prior(gaussian_prior, hyp['output_scale_mean'], hyp['output_scale_var'])]
            self.fun = isotropic_matern_covariance
            self.priors = priors
        elif name == 'PER':
            priors = [get_prior(gaussian_prior, hyp['p_length_scale_mean'], hyp['p_length_scale_var']),
                      get_prior(gaussian_prior, hyp['p_mean'], hyp['p_var']),
                      get_prior(gaussian_prior, hyp['output_scale_mean'], hyp['output_scale_var'])]
            self.fun = periodic_covariance
            self.priors = priors
        elif name == 'LIN':
            priors = [get_prior(gaussian_prior, hyp['output_scale_mean'], hyp['output_scale_var'])]
            self.fun = linear_covariance
            self.priors = priors
        elif name == 'RQ':
            priors = [get_prior(gaussian_prior, hyp['length_scale_mean'], hyp['length_scale_var']),
                      get_prior(gaussian_prior, hyp['output_scale_mean'], hyp['output_scale_var']),
                      get_prior(gaussian_prior, hyp['alpha_mean'], hyp['alpha_var'])]
            self.fun = isotropic_rq_covariance
            self.priors = priors

        self.fixed_hyps = None
