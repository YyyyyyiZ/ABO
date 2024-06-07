from util.problem import Covariance

import numpy as np


def covMask(cov, hyp, x, z=None, i=None):
    mask = cov[0].flatten().astype(int)  # either a binary mask or an index set
    cov = cov[1]  # covariance function to be masked
    if isinstance(cov[0], list):  # properly unwrap nested cell arrays
        cov = cov[0]
    nh_string = cov()  # number of hyperparameters of the full covariance

    if max(mask) < 2 and len(mask) > 1:
        mask = list(np.where(mask)[0])  # convert 1/0->index
    D = len(mask)  # masked dimension

    if z is None:
        xeqz = True
        dg = False
    else:
        xeqz = False
        dg = z == 'diag'  # determine mode

    if i is None:
        return str(eval(nh_string))  # number of parameters

    if dg:
        if xeqz:
            return cov(hyp, x[:, mask], 'diag', i)
        else:
            return cov(hyp, x[:, mask], z[:, mask], 'diag', i)
    else:
        if xeqz:
            return cov(hyp, x[:, mask], i)
        else:
            return cov(hyp, x[:, mask], z[:, mask], i)


def mask_covariance(K, theta, x, z=None, i=None, j=None):
    if K is None:
        raise ValueError('Missing Argument', 'covariance input K is required!')
    elif z is None and i is None:
        return covMask(K)
    elif z is None and i is not None:
        return covMask(K, theta, x)
    elif i is not None and j is None:
        return covMask(K, theta, x, z)
    else:
        mask = K[0].flatten().astype(int)
        K = K[1]
        if isinstance(K[0], list):  # expand cell
            K = K[0]

        if z is None:
            return K(theta, x[:, mask], None, i, j)
        else:
            return K(theta, x[:, mask], z[:, mask], i, j)


def mask_kernels(covariances, num_feature):
    new_set_of_covariances = []
    new_set_of_names = []
    for i in range(len(covariances)):
        for j in range(num_feature):
            mask_features = j
            new_cov = Covariance()
            new_cov.fun = mask_covariance(mask_features, covariances[i].fun)
            new_cov.priors = covariances[i].priors
            new_set_of_covariances.append(new_cov)
            new_set_of_names.append(covariances[i].cov_name)
    return
