import numpy as np
import random
import GPy
from GPy.kern.src.rbf import RBF

from abo import boms_hyperparameters
from gp.gp_prior import get_prior, gaussian_prior
from util.tools import prior_sample, hellinger_distance


def create_bo_prior(x, samples, noise, data_subsample_size):
    prior = {}

    prior['n'] = 0
    prior['K'] = np.zeros((0, 0))
    prior['num_hyps'] = []
    prior['candidates'] = []
    prior['evaluated'] = []
    prior['covariances'] = []
    prior['names'] = []
    ind = random.sample(range(x.shape[0]), min(x.shape[0], data_subsample_size))
    prior['x'] = x[ind, :]
    prior['samples'] = samples
    prior['cov_matrices'] = []

    hyps = boms_hyperparameters(noise)

    # mean_function = GPy.mappings.Constant(input_dim=1, output_dim=1, value=0.5)
    mean_priors = [get_prior(gaussian_prior, hyps['mean_offset'], hyps['mean_var'])]
    prior['mean'] = {'priors': mean_priors, 'fun': [GPy.mappings.Constant]}

    cov_priors = [get_prior(gaussian_prior, hyps['model_length_scale_mean'], hyps['model_length_scale_var']),
                  get_prior(gaussian_prior, hyps['output_scale_mean'], hyps['output_scale_var'])]
    prior['cov'] = {'priors': cov_priors, 'fun': [RBF]}

    prior['noise_prior'] = [get_prior(gaussian_prior, hyps['lik_noise_std'], hyps['lik_noise_std_var'])]

    return prior


def update_bo_prior(prior, new_covariances=None, new_names=None, new_evaluated=None, removed_candidates=None):
    if new_covariances is None:
        new_covariances = []
    if new_names is None:
        new_names = []
    if new_evaluated is None:
        new_evaluated = []
    if removed_candidates is None:
        removed_candidates = []

    if len(new_covariances) != len(new_names):
        raise ValueError('# of new covs must match # of new names')

    prior['candidates'] = [c for c in prior['candidates'] if c not in removed_candidates]

    n_new = len(new_covariances)

    if n_new > 0:
        n = prior['n']
        prior['candidates'].extend(range(n + 1, n + n_new + 1))
        prior['covariances'].extend(new_covariances)
        prior['names'].extend(new_names)

        K = np.ones((n + n_new, n + n_new)) - np.eye(n + n_new)
        K[:n, :n] = prior['K']
        prior['K'] = K
        prior['n'] = n + n_new

    covariances = prior['covariances']
    samples = prior['samples']
    x = prior['x']
    noise_prior = prior['noise_prior']
    noise_samples = np.exp(prior_sample(noise_prior, samples[:, -1]))

    if n_new > 0:
        for i in range(n + 1, n + n_new + 1):
            saved_covs = []
            cov = covariances[i - 1]
            hyps = prior_sample(cov['priors'], samples.shape)
            for s in range(samples.shape[0]):
                k = cov['fun'][0](prior['K'], hyps[s, :])
                k = k + noise_samples[s] * np.eye(k.shape[0])
                saved_covs.append(k)
            prior['cov_matrices'].append(saved_covs)

    if n_new > 0:
        for i in range(n + 1, n + n_new + 1):
            for j in prior['evaluated']:
                alignment = compute_alignment(prior['cov_matrices'][i - 1], prior['cov_matrices'][j - 1])
                prior['K'][i - 1, j - 1] = alignment
                prior['K'][j - 1, i - 1] = alignment

    if new_evaluated:
        prior['evaluated'].extend(new_evaluated)
        prior['candidates'] = [c for c in prior['candidates'] if c not in new_evaluated]
        for i in new_evaluated:
            for j in new_evaluated:
                if i < j:
                    alignment = compute_alignment(prior['cov_matrices'][i - 1], prior['cov_matrices'][j - 1])
                    prior['K'][i - 1, j - 1] = alignment
                    prior['K'][j - 1, i - 1] = alignment

            for j in prior['candidates']:
                alignment = compute_alignment(prior['cov_matrices'][i - 1], prior['cov_matrices'][j - 1])
                prior['K'][i - 1, j - 1] = alignment
                prior['K'][j - 1, i - 1] = alignment

    num_eval = len(prior['evaluated'])
    num_cand = len(prior['candidates'])
    K = np.ones((num_eval + num_cand, num_eval + num_cand)) - np.eye(num_eval + num_cand)
    K[:num_eval, :num_eval] = prior['K'][prior['evaluated'], :][:, prior['evaluated']]
    K[:num_eval, num_eval:] = prior['K'][prior['evaluated'], :][:, prior['candidates']]
    K[num_eval:, :num_eval] = prior['K'][prior['candidates'], :][:, prior['evaluated']]

    covariances = [prior['covariances'][i - 1] for i in prior['evaluated']]
    covariances.extend([prior['covariances'][i - 1] for i in prior['candidates']])

    names = [prior['names'][i - 1] for i in prior['evaluated']]
    names.extend([prior['names'][i - 1] for i in prior['candidates']])

    cov_matrices = [prior['cov_matrices'][i - 1] for i in prior['evaluated']]
    cov_matrices.extend([prior['cov_matrices'][i - 1] for i in prior['candidates']])

    prior['covariances'] = covariances
    prior['names'] = names
    prior['cov_matrices'] = cov_matrices
    prior['evaluated'] = list(range(1, num_eval + 1))
    prior['candidates'] = list(range(num_eval + 1, num_eval + num_cand + 1))
    prior['K'] = K
    prior['n'] = num_eval + num_cand

    return prior


def compute_alignment(covs1, covs2):
    alignment = 0
    n = len(covs1)
    for s in range(n):
        k1 = covs1[s]
        k2 = covs2[s]
        m = np.zeros(k1.shape[0])
        alignment += hellinger_distance(m, k1, m, k2)
    alignment /= n
    return alignment
