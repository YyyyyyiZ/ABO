import numpy as np


def prod_covariance(cov, hyp=None, x=None, z=None, i=None, j=None):
    if hyp is None:
        return covProd(cov)
    elif x is None:
        return covProd(cov, hyp)
    elif z is None:
        return covProd(cov, hyp, x)
    elif i is None:
        return covProd(cov, hyp, x, z)
    elif j is None:
        return covProd(cov, hyp, x, z, i)
    else:
        if i > j:
            return prod_covariance(cov, hyp, x, z, j, i)

        nh = []
        for f in cov:  # iterate over covariance functions
            if isinstance(f(), list):
                f = f[0]
            nh.append(str(eval(f.__name__ + '()')))

        v = []
        for ii in range(len(cov)):
            v.extend([ii + 1] * eval(nh[ii]))

        if j <= len(v):
            K = 1
            vi = v[i - 1]  # MATLAB indexing starts from 1, Python from 0
            vj = v[j - 1]
            ki = sum(1 for k in v[:i] if k == vi)
            kj = sum(1 for k in v[:j] if k == vj)

            for ii in range(len(cov)):  # iteration over factor functions
                f = cov[ii]
                if isinstance(f(), list):
                    f = f[0]
                if vi == vj:
                    if ii + 1 == vj:
                        K *= f(hyp[v == ii + 1], x, z, ki, kj)
                    else:
                        K *= f(hyp[v == ii + 1], x, z)
                else:
                    if ii + 1 == vi:
                        K *= f(hyp[v == ii + 1], x, z, ki)
                    else:
                        K *= f(hyp[v == ii + 1], x, z, kj)
            return K
        else:
            raise ValueError('Unknown hyperparameter')


def covProd(cov, hyp=None, x=None, z=None, i=None):
    if not cov:
        raise ValueError('We require at least one factor.')

    j = []
    for f in cov:
        if isinstance(f(), list):
            f = f[0]
        j.append(str(f.__name__))

    if hyp is None:
        return '+'.join(j)

    if z is None:
        z = np.array([])

    n, D = x.shape

    v = []
    for ii in range(len(cov)):
        v.extend([ii + 1] * eval(j[ii]))

    if i is None:
        K = 1
        for ii in range(len(cov)):
            f = cov[ii]
            if isinstance(f(), list):
                f = f[0]
            K *= f(hyp[np.array(v) == ii + 1], x, z)
        return K
    else:
        if i <= len(v):
            K = 1
            vi = v[i - 1]
            j = sum(1 for k in v[:i] if k == vi)
            for ii in range(len(cov)):
                f = cov[ii]
                if isinstance(f(), list):
                    f = f[0]
                if ii + 1 == vi:
                    K *= f(hyp[np.array(v) == ii + 1], x, z, j)
                else:
                    K *= f(hyp[np.array(v) == ii + 1], x, z)
            return K
        else:
            raise ValueError('Unknown hyperparameter')


def sum_covariance(cov, hyp=None, x=None, z=None, i=None, j=None):
    if hyp is None:
        return covSum(cov)
    elif x is None:
        return covSum(cov, hyp)
    elif z is None:
        return covSum(cov, hyp, x)
    elif i is None:
        return covSum(cov, hyp, x, z)
    elif j is None:
        return covSum(cov, hyp, x, z, i)
    else:
        if i > j:
            return sum_covariance(cov, hyp, x, z, j, i)

        nh = []
        for f in cov:
            if isinstance(f(), list):
                f = f[0]
            nh.append(str(eval(f.__name__ + '()')))

        v = []
        for ii in range(len(cov)):
            v.extend([ii + 1] * eval(nh[ii]))

        if j <= len(v):
            vi = v[i - 1]
            vj = v[j - 1]
            ki = sum(1 for k in v[:i] if k == vi)
            kj = sum(1 for k in v[:j] if k == vj)

            if vi != vj:
                return np.zeros(x.shape[0])

            f = cov[vj - 1]
            if isinstance(f(), list):
                f = f[0]
            return f(hyp[np.array(v) == vj], x, z, ki, kj)
        else:
            raise ValueError('Unknown hyperparameter')


def covSum(cov, hyp=None, x=None, z=None, i=None):
    if not cov:
        raise ValueError('We require at least one summand.')

    j = []
    for f in cov:
        if isinstance(f(), list):
            f = f[0]
        j.append(str(f.__name__))

    if hyp is None:
        return '+'.join(j)

    if z is None:
        z = np.array([])

    n, D = x.shape

    v = []
    for ii in range(len(cov)):
        v.extend([ii + 1] * eval(j[ii]))

    if i is None:
        K = 0
        for ii in range(len(cov)):
            f = cov[ii]
            if isinstance(f(), list):
                f = f[0]
            K += f(hyp[np.array(v) == ii + 1], x, z)
        return K
    else:
        if i <= len(v):
            vi = v[i - 1]
            j = sum(1 for k in v[:i] if k == vi)
            f = cov[vi - 1]
            if isinstance(f(), list):
                f = f[0]
            return f(hyp[np.array(v) == vi], x, z, j)
        else:
            raise ValueError('Unknown hyperparameter')
