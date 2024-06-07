from util.problem import Covariance
from covariance.operator import sum_covariance, prod_covariance
from bo.candidates import remove_duplicate_candidates

import numpy as np


def covariance_grammar_started(covariances_names, hyp, d):
    covariances = []
    for i in range(len(covariances_names)):
        m = Covariance(covariances_names[i], hyp, d)
        covariances.append(m)
    return


def get_next_covariances(cov, base_covs):
    return get_next_covariances_recur(cov, base_covs, ' ')


def get_next_covariances_recur(cov, base_covs, mode):
    """
    Find next kernels by:
    - Replacing S with S + B or S * B, where S is a subexpression of kernel and B is a base kernel.
    - Replacing any B with B', where B' is another base kernel.
    """
    covs = []
    # Append new kernel to existing kernel
    for base_cov in base_covs:
        if mode != '+':
            covs.append(combine_tokens('+', base_cov, cov))
        if mode != '*':
            covs.append(combine_tokens('*', base_cov, cov))

    is_base = False
    '''
    if all(size(cov.fun) == [1,1])
        isBase = true;
    '''
    if len(cov.fun) == 1:
        is_base = True
    else:
        op, _ = cov.fun
        if op.__name__ == 'mask_covariance':
            is_base = True

    # If kernel is a base kernel, replace it with the other base kernels
    if is_base:
        for base_cov in base_covs:
            if base_cov.fun != cov.fun:
                covs.append(base_cov)
    else:
        # If kernel is a composition of kernels, recur on sub-expressions
        op, args = cov.fun
        # if op in [prod_covariance_fast, sum_covariance_fast]:
        #     args = args[0]

        # num_hyp = eval(feval(args{1}{:}))
        num_hyp = eval(args[0])

        token_1 = Covariance()
        token_2 = Covariance()
        token_1.fun = args[0]
        token_2.fun = args[1]

        if num_hyp == 0:
            token_1.priors = []
            token_2.priors = []
        else:
            token_1.priors = cov.priors[:num_hyp]
            token_2.priors = cov.priors[num_hyp:]

        # Recur on first argument
        if op in [prod_covariance]:
            next_candidates = get_next_covariances_recur(token_1, base_covs, '*')
            for nc in next_candidates:
                covs.append(combine_tokens('*', nc, token_2))
        elif op in [sum_covariance]:
            next_candidates = get_next_covariances_recur(token_1, base_covs, '+')
            for nc in next_candidates:
                covs.append(combine_tokens('+', nc, token_2))
        else:
            raise ValueError('Operation not defined.')

        # If arguments are different, recur on second argument
        if args[0] != args[1]:
            if op in [prod_covariance]:
                next_candidates = get_next_covariances_recur(token_2, base_covs, '*')
                for nc in next_candidates:
                    covs.append(combine_tokens('*', token_1, nc))
            elif op in [sum_covariance]:
                next_candidates = get_next_covariances_recur(token_2, base_covs, '+')
                for nc in next_candidates:
                    covs.append(combine_tokens('+', token_1, nc))
            else:
                raise ValueError('Operation not defined.')

    return covs


def expand_covariance(cov, base_covs, base_names, max_depth):
    count = 0
    name = covariance2str(cov.fun)

    for base_name in base_names:
        count += name.count(base_name)

    if count > max_depth:
        covs = []
    elif count == max_depth:
        covs = expand_covs_recur(cov, base_covs)
    else:
        covs = get_next_covariances(cov, base_covs)
        covs.extend(expand_covs_recur(cov, base_covs))
    return covs


def expand_covs_recur(cov, base_covs):
    covs = []
    # If kernel is a base kernel, return
    if len(cov.fun) == 1:
        return covs
    op, args = cov.fun[0], cov.fun[1:]
    if op.__name__ == 'mask_covariance':
        return covs

    # If kernel is a composition of kernels, recur on sub-expressions.
    # if op in [prod_covariance_fast, sum_covariance_fast]:
    #     args = args[0]

    num_hyp = eval(args[0])

    token_1 = Covariance()
    token_2 = Covariance()
    token_1.fun = args[0]
    token_2.fun = args[1]

    if num_hyp == 0:
        token_1.priors = []
        token_2.priors = []
    else:
        token_1.priors = cov.priors[:num_hyp]
        token_2.priors = cov.priors[num_hyp:]

    # switch operator
    if op in [prod_covariance]:
        covs.append(combine_tokens('*', token_1, token_2))
    elif op in [sum_covariance]:
        covs.append(combine_tokens('+', token_1, token_2))
    else:
        raise ValueError('Operation not defined')

    # remove a base kernel
    '''
    if all(size(token_1.fun) == [1,1])
        covs = [covs, token_2];
    end
    if all(size(token_2.fun) == [1,1])
        covs = [covs, token_1];
    end
    '''
    if len(token_1.fun) == 1:
        covs.append(token_2)
    if len(token_2.fun) == 1:
        covs.append(token_1)

    # recur on arguments
    new_covs_1 = expand_covs_recur(token_1, base_covs)
    new_covs_2 = expand_covs_recur(token_2, base_covs)

    if op in [prod_covariance]:
        for nc in new_covs_1:
            covs.append(combine_tokens('*', nc, token_2))
        for nc in new_covs_2:
            covs.append(combine_tokens('*', token_1, nc))
    elif op in [sum_covariance]:
        for nc in new_covs_1:
            covs.append(combine_tokens('+', nc, token_2))
        for nc in new_covs_2:
            covs.append(combine_tokens('+', token_1, nc))
    else:
        raise ValueError('Operation not defined')

    return covs


def fully_expand_tree(base_covs, depth, max_number_models):
    covs = base_covs
    base_names = []
    # get names of base covariances
    for i in range(len(base_covs)):
        base_names.append(base_covs[i].covariance_name)
    covs_to_expand = covs
    names = base_names
    level_sizes = np.zeros((1, depth + 1))
    # expand tree
    for i in range(depth):
        new_covs = []
        for j in range(len(covs_to_expand)):
            temp_covs = expand_covariance(covs_to_expand[j], base_covs, base_names, depth + 2)
            temp_covs, new_names = remove_duplicate_candidates(temp_covs, names, base_names)
            names = names + new_names
            new_covs = new_covs + temp_covs
            if len(covs) + len(new_covs) >= max_number_models:
                break
        covs = covs + new_covs
        covs_to_expand = new_covs
        level_sizes.append(len(covs))
        if len(covs) >= max_number_models:
            break
    return covs, names, level_sizes


def combine_tokens(op, m1, m2):
    result = [[m1.fun, m2.fun], precompute_stuff([m1.fun, m2.fun])]
    m = Covariance()
    if op == '+':
        m.fun = sum_covariance(result)
    elif op == '*':
        m.fun = prod_covariance(result)
    else:
        raise ValueError('Unknown operation')

    m.priors = [m1.priors, m2.priors]

    if m1.fixed_hyps is not None and m2.fixed_hyps is not None:
        m.fixed_hyps = np.concatenate((m1.fixed_hyps, m2.fixed_hyps))
    return m


def precompute_stuff(cov):
    j = []
    for f in cov:  # iterate over covariance functions
        if isinstance(f, list):
            f = f[0]
        j.append(str(f()))
    v = []
    for ii in range(len(cov)):
        v.extend([ii + 1] * eval(j[ii]))

    return [j, v]


def covariance2str(covariance_handle):
    # cov_handle = [sum_covariance, [[isotropic_sqdexp_covariance], [periodic_covariance]]]
    # Output: (SE+PER)
    if isinstance(covariance_handle, list) and len(covariance_handle) > 1:
        fname = covariance_handle[0].__name__
    else:
        if isinstance(covariance_handle, list):
            fname = covariance_handle[0].__name__
        else:
            fname = covariance_handle.__name__

    if fname == 'isotropic_sqdexp_covariance':
        covariance_name = 'SE'
    elif fname == 'isotropic_matern_covariance':
        covariance_name = 'M1'
    elif fname == 'periodic_covariance':
        covariance_name = 'PER'
    elif fname == 'linear_covariance':
        covariance_name = 'LIN'
    elif fname == 'isotropic_rq_covariance':
        covariance_name = 'RQ'
    elif fname == 'ard_sqdexp_covariance':
        covariance_name = 'SEard'
    elif fname == 'sum_covariance':
        covariance_name = f"({covariance2str(covariance_handle[1][0])}+{covariance2str(covariance_handle[1][1])})"
    elif fname == 'prod_covariance':
        covariance_name = f"({covariance2str(covariance_handle[1][0])}*{covariance2str(covariance_handle[1][1])})"
    elif fname == 'sum_covariance_fast':
        covariance_name = f"({covariance2str(covariance_handle[1][0][0])}+{covariance2str(covariance_handle[1][0][1])})"
    elif fname == 'prod_covariance_fast':
        covariance_name = f"({covariance2str(covariance_handle[1][0][0])}*{covariance2str(covariance_handle[1][0][1])})"
    elif fname == 'mask_covariance':
        covariance_name = covariance2str(covariance_handle[1][1])
        index = str(covariance_handle[1][0])
        covariance_name = f"{covariance_name}_{index}"
    else:
        covariance_name = 'Unk'

    return covariance_name

