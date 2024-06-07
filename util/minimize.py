import numpy as np
from scipy.optimize import minimize
from gpml_extensions import unwrap, rewrap, gp_optimizer_wrapper, fix_pd_matrix


def minimize_minFunc(model, x, y, initial_hyperparameters=None, num_restarts=3, minFunc_options=None):
    if minFunc_options is None:
        minFunc_options = {
            'disp': False,
            'maxiter': 500,
            'method': 'L-BFGS-B'
        }

    if initial_hyperparameters is None:
        initial_hyperparameters = model['prior']()

    f = lambda hyperparameter_values: gp_optimizer_wrapper(
        hyperparameter_values,
        initial_hyperparameters,
        model['inference_method'],
        model['mean_function'],
        model['covariance_function'],
        model['likelihood'],
        x, y
    )

    best_hyperparameter_values, best_nlZ, exitflag, best_minFunc_output = \
        run_minFunc(f, unwrap(initial_hyperparameters), minFunc_options)

    if exitflag < 0:
        print('MinFunc failed trying to optimize initial_hyperparameters. Random restart')
        num_restarts += 1
        best_nlZ = np.inf

    for i in range(num_restarts):
        hyperparameters = model['prior']()

        hyperparameter_values, nlZ, exitflag, minFunc_output = \
            run_minFunc(f, unwrap(hyperparameters), minFunc_options)

        try:
            theta = rewrap(initial_hyperparameters, hyperparameter_values)
            K = model['covariance_function'](theta['cov'], x)
            L = np.linalg.cholesky(fix_pd_matrix(K))
        except:
            nlZ = np.inf
            exitflag = -1

        if exitflag < 0:
            for j in range(10):
                hyperparameters = model['prior']()

                print(f'MinFunc failed trying to optimize the hyperparameters. N# failures is {j + 1} out of 10')
                hyperparameter_values, nlZ, exitflag, minFunc_output = \
                    run_minFunc(f, unwrap(hyperparameters), minFunc_options)

                try:
                    theta = rewrap(initial_hyperparameters, hyperparameter_values)
                    K = model['covariance_function'](theta['cov'], x)
                    L = np.linalg.cholesky(fix_pd_matrix(K))
                except:
                    nlZ = np.inf
                    exitflag = -1

                if exitflag > 0:
                    break

        if (nlZ < best_nlZ and abs(nlZ - best_nlZ) > 1e-6):
            best_nlZ = nlZ
            best_hyperparameter_values = hyperparameter_values
            best_minFunc_output = minFunc_output

    best_hyperparameters = rewrap(initial_hyperparameters, best_hyperparameter_values)

    if np.isnan(best_nlZ) or np.isinf(best_nlZ):
        raise RuntimeError('Optimization failed')

    return best_hyperparameters, best_nlZ, best_minFunc_output


def run_minFunc(f, initial_values, minFunc_options):
    result = minimize(f, initial_values, **minFunc_options)
    return result.x, result.fun, result.status, result
