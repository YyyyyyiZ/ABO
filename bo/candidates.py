import numpy as np
import random
import re
from scipy.stats import geom, norm


from grammar.covariance_grammar import covariance_grammar_started, get_next_covariances
from grammar.mask_kernels import mask_kernels


def select_new_candidates(problem, explore_budget, exploit_budget, names, neighborhoods, starting_depth):
    """
    Select new candidates for exploration and exploitation.

    Parameters:
    problem (object): Problem definition containing theta_models and d.
    explore_budget (int): Budget for exploration.
    exploit_budget (int): Budget for exploitation.
    names (list): List of current candidate names.
    neighborhoods (list): List of neighborhood candidates.
    starting_depth (int): Starting depth for exploration.

    Returns:
    new_covs (list): List of new covariance functions.
    new_names (list): List of new candidate names.
    """
    new_covs = []
    new_names = []
    depth_prob = 1 / 2

    # Exploration
    base_cov_masked, base_cov_masked_names = mask_kernels(
        covariance_grammar_started(['SE', 'RQ'], problem.theta_models, problem.d), problem.d)

    while len(new_covs) < explore_budget:
        depth = min(10, starting_depth + geom.pmf(1, depth_prob))
        cov = random.choice(base_cov_masked)
        for i in range(int(depth) - 1):
            covs = get_next_covariances(cov, base_cov_masked)
            cov = random.choice(covs)

        cov, name = remove_duplicate_candidates([cov], names + new_names, base_cov_masked_names)
        if len(cov) > 0:
            new_covs.append(cov[0])
            new_names.append(name[0])

    # Exploitation
    if exploit_budget > 0:
        for i in range(len(neighborhoods)):
            if len(neighborhoods[i]) == 0:
                break
            n_neighborhoods_to_sample = min(exploit_budget, len(neighborhoods[i]))
            covs = random.sample(neighborhoods[i], n_neighborhoods_to_sample)
            covs, nnames = remove_duplicate_candidates(covs, names + new_names, base_cov_masked_names)
            if len(covs) > 0:
                new_covs.extend(covs)
                new_names.extend(nnames)

    return new_covs, new_names


def remove_duplicate_candidates(possible_candidates, used_names, base_cov_names):
    candidates = []
    new_names = []
    names = used_names

    for i in range(len(possible_candidates)):
        valid = 1
        name = possible_candidates[i].covariance_name
        for j in range(len(names)):
            name_2 = names[j]
            counts_1 = np.zeros((5, 1))
            counts_2 = np.zeros((5, 1))
            # check if a candidate has the same combination of base
            # covariances as an already used covariance.
            counts_1 = [0] * len(base_cov_names)
            counts_2 = [0] * len(base_cov_names)
            for k in range(len(base_cov_names)):
                counts_1[k] = len(re.findall(base_cov_names[k], name))
                counts_2[k] = len(re.findall(base_cov_names[k], name_2))
            if counts_1 != counts_2:
                continue
            # if so, check to see if they have the same algebraic expression.
            equal = 1
            for k in range(1, 3):
                SE = random.random()
                RQ = random.random()
                M1 = random.random()
                LIN = random.random()
                PER = random.random()
                SEard = random.random()

                name_exp = name.replace('_', '*')
                name2_exp = names[j].replace('_', '*')
                if abs(eval(name_exp) - eval(name2_exp)) > 1e-6:
                    equal = 0
                    break
            # if so, this candidate is a duplicate and can be discarded
            if equal:
                valid = 0
                break
        if valid:
            candidates.append(possible_candidates[i])
            names = names + name
            new_names = new_names + name

    return candidates, new_names


def select_best_candidate(best_y, x_star, mu_star, s2_star, times):
    """
    Select the next test point using Expected Improvement (EI).

    Parameters:
    best_y (float): Best observed value so far.
    x_star (ndarray): Array of candidate test points.
    mu_star (ndarray): Array of mean predictions at the candidate points.
    s2_star (ndarray): Array of variance predictions at the candidate points.
    times (ndarray): Array of exploration times for each candidate.

    Returns:
    next_x (float): Next test point selected using EI.
    scores (ndarray): Array of EI scores for each candidate point.
    """
    best_ei = float('-inf')
    scores = np.zeros(len(x_star))

    for i in range(len(x_star)):
        ei = compute_ei(best_y, mu_star[i], np.sqrt(s2_star[i])) / times[i]
        scores[i] = ei

        if ei >= best_ei:
            best_ei = ei
            next_x = x_star[i]

    return next_x, scores


def compute_ei(y_max, mu, sigma):
    """
    Compute the Expected Improvement (EI) given the best observed value, mean prediction, and standard deviation.

    Parameters:
    y_max (float): Best observed value so far.
    mu (float): Mean prediction at the candidate point.
    sigma (float): Standard deviation prediction at the candidate point.

    Returns:
    ei (float): Expected Improvement (EI) score.
    """
    sigma[sigma < 0] = 0

    # Compute expected improvement
    delta = mu - y_max
    u = delta / sigma
    u_pdf = norm.pdf(u)
    u_cdf = norm.cdf(u)

    ei = delta * u_cdf + sigma * u_pdf
    ei[ei < 0] = 0

    return ei


