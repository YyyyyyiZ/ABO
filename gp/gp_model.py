import time
import GPy
import sys
import random

from grammar.covariance_grammar import *
from grammar.mask_kernels import mask_kernels
from util.problem import Theta
from scipy import norm


def get_initial_models(problem, d, data_noise):
    theta = Theta(data_noise)
    # incomplete
    covariances_root = covariance_grammar_started(['SE', 'RQ'], theta, d)
    base_cov_masked = mask_kernels(covariances_root, d)
    covariances = fully_expand_tree(base_cov_masked, 1, problem.max_candidates)

    if d > 2:
        models_d1 = covariances[2 * d + 1:]
        covariances = covariances[:2 * d + 1]
        random_index = random.sample(range(len(models_d1)), d * 4)
        for item in random_index:
            covariances.append(models_d1[item])
        fully_additive = covariances[0]

        # Just for SE
        for i in range(1, d+1):
            fully_additive = combine_tokens('+', fully_additive, covariances[i])
        covariances.append(fully_additive.fun)

        # Just for RQ
        fully_additive = covariances[d]
        for i in range(1, d+1):
            fully_additive = combine_tokens('+', fully_additive, covariances[i])
        covariances.append(fully_additive.fun)

    covariances.append(covariance_grammar_started(['SEard'], theta, d))
    models = []
    for i in range(len(covariances)):
        models.append(gpr_model_builder(problem, covariances[i], theta))
        print(covariance2str(covariances[i].fun))
    return models


def gp_train(x, y, model):
    model.set_XY(x, y)
    try:
        model.optimize(optimizer='lbfgs', maxiter=100)
    except:
        print('minFunc failed')
        sys.exit(1)
        # need to restart

    return model


def gp_update(problem, models, x, y):
    log_evidence = []

    for m, model in enumerate(models):
        start_time = time.time()
        try:
            model.parameters = gp_train(x, y, model['parameters'])
        except:
            print(f'Error during training model {m}. Removing points that are too close to each other.\n')
            n = x.shape[0]
            exclude = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(x[i] - x[j]) < 0.05:
                        exclude[i] = True
            x = x[~exclude]
            y = y[~exclude]
            model.parameters = gp_train(x, y, model.parameters)
        current_log_evidence = gp_log_evidence(model.parameters)
        log_evidence.append(current_log_evidence)
        model['log_evidence'] = current_log_evidence
        end_time = time.time()
        print('Model {} trained in {.3f} s\n'.format(m, end_time - start_time))
    return models, log_evidence


def gpr_model_builder(covariance_model, hyper_param):
    # Likelihood specification
    likelihood = GPy.likelihoods.Gaussian(variance=hyper_param.lik_noise_std ** 2)
    # Mean function specification
    mean_function = GPy.mappings.constant.Constant(input_dim=1, output_dim=1, value=hyper_param.mean_offset)
    # Covariance function specifications
    cov_function = covariance_model.fun
    # Creating the GP model
    gp_model = GPy.core.GP(X=None, Y=None, kernel=cov_function, likelihood=likelihood,
                           mean_function=mean_function)
    # Priors (using Gaussian priors as an example, similar to GPML toolbox)
    gp_model.mean_function.set_prior(GPy.priors.Gaussian(hyper_param.mean_offset, np.sqrt(hyper_param.mean_var)))
    gp_model.likelihood.variance.set_prior(
        GPy.priors.Gaussian(hyper_param.lik_noise_std, np.sqrt(hyper_param.lik_noise_std_var)))

    for i, prior in enumerate(covariance_model.priors):
        gp_model.kern.parameters[i].set_prior(GPy.priors.Gaussian(prior[0], np.sqrt(prior[1])))

    model = {
        'parameters': gp_model,
        'log_evidence': None,
        # 'update': gp_update,
        # 'entropy': gpr_entropy,
    }
    return model


def gp_log_evidence(model):
    d = model.kern.input_dim
    half_log_2pi = 0.918938533204673
    L_diag = np.diag(model.posterior.woodbury_chol)
    log_evidence = -model.log_likelihood() + d * half_log_2pi - np.sum(np.log(L_diag))
    return log_evidence


def gp(hyp, inf, mean, cov, lik, x, y, xs=None, ys=None):
    """
    Gaussian Process inference and prediction.

    Parameters:
    hyp: dict of hyperparameters
    inf: inference method
    mean: mean function
    cov: covariance function
    lik: likelihood function
    x: training inputs (n by D)
    y: training targets (n by 1)
    xs: test inputs (ns by D)
    ys: test targets (ns by 1)

    Returns:
    Training mode:
    nlZ: negative log marginal likelihood
    dnlZ: derivatives of nlZ w.r.t hyperparameters
    post: posterior
    Prediction mode:
    ymu: predictive output means
    ys2: predictive output variances
    fmu: predictive latent means
    fs2: predictive latent variances
    lp: log predictive probabilities
    post: posterior
    """

    # Set default mean function
    if mean is None:
        mean = GPy.mappings.constant.Constant(input_dim=x.shape[1], value=0.0)

    # Define the GP model
    kernel = cov(hyp['cov'])
    if lik == 'Gaussian':
        likelihood = GPy.likelihoods.Gaussian(variance=hyp['lik'][0])
    else:
        raise ValueError('Unsupported likelihood function')

    model = GPy.core.GP(X=x, Y=y, kernel=kernel, likelihood=likelihood, mean_function=mean)

    # Training mode
    if xs is None:
        nlZ = -model.log_likelihood()
        dnlZ = model.gradients.copy()
        return nlZ, dnlZ, model

    # Prediction mode
    else:
        ymu, ys2 = model.predict(xs, full_cov=False)
        fmu, fs2 = model.predict_f(xs, full_cov=False)

        if ys is not None:
            # Compute log predictive probabilities
            lp = norm.logpdf(ys, loc=ymu, scale=np.sqrt(ys2))
            return ymu, ys2, fmu, fs2, lp, model
        else:
            return ymu, ys2, fmu, fs2, None, model


# # Define helper functions for covariance and mean functions
# def rbf_kernel(hyp_cov):
#     return GPy.kern.RBF(input_dim=len(hyp_cov), variance=hyp_cov[0], lengthscale=hyp_cov[1:])
#
#
# def constant_mean(input_dim, value):
#     return GPy.mappings.constant.Constant(input_dim=input_dim, value=value)
#
# # Example usage
# hyp = {
#     'cov': [1.0, 1.0],
#     'lik': [1.0],
#     'mean': []
# }
# x = np.random.rand(10, 1)
# y = np.sin(x)
#
# # Train the model
# nlZ, dnlZ, model = gp(hyp, inf='exact', mean=constant_mean(1, 0), cov=rbf_kernel, lik='Gaussian', x=x, y=y)
#
# # Make predictions
# xs = np.linspace(0, 1, 100)[:, None]
# ymu, ys2, fmu, fs2, lp, model = gp(hyp, inf='exact', mean=constant_mean(1, 0), cov=rbf_kernel, lik='Gaussian', x=x, y=y,
#                                    xs=xs)

