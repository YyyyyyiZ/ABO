from gp.gp_model import *
from query_strategy.acq_function import expected_improvement_limited_optimization
from util.problem import Theta
from util.tools import sobol_sample, get_num_hyps
from grammar.covariance_grammar import *
from grammar.mask_kernels import *
from bo.bo import create_bo_prior, update_bo_prior
from bo.candidates import remove_duplicate_candidates, select_new_candidates, select_best_candidate


def abo(problem, models, query_strategy):
    x = problem.initial_x
    y = problem.initial_y
    x_star = np.empty(problem.budget, problem.d)
    y_star = np.empty(problem.budget, 1)

    # Updating initial bag of models
    models, log_evidence = gp_update(problem, models, x, y)
    boms_models = models

    # candidate points
    theta = Theta(problem.data_noise)
    covariances_root = problem.covariance_root
    covariances = covariance_grammar_started(covariances_root, theta, problem.d)

    for i in range(len(covariances)):
        covariances[i].fixed_hyps = np.full((len(covariances[i].priors), 1), float('inf'))
    if problem.d > 1:
        covariances = mask_kernels(covariances, problem.d)

    base_names = []
    for i in range(len(covariances)):
        base_names.append(covariance2str(covariances[i].fun))

    starting_depth = 2
    max_candidates = problem.max_candidates
    covariances = fully_expand_tree(covariances, starting_depth, max_candidates)

    # initial models
    init_models = boms_models
    init_cov = []
    init_names = []
    for i in range(len(init_models)):
        temp_cov = Covariance()
        temp_cov.fun = init_models[i].parameters.covariance_function
        temp_cov.priors = init_models[i].parameters.priors.cov
        init_cov.append(temp_cov)
        init_names.append(covariance2str(init_cov[i].fun))

    covariances = init_cov
    new_cov, new_names = remove_duplicate_candidates(covariances, init_names, base_names)
    names = init_names
    init_num_models = len(init_cov)
    for i in range(new_names):
        init_cov.append(new_cov[i])
        init_names.append(new_names[i])

    samples = sobol_sample(problem.total_hyp_samples, 300)
    prior = create_bo_prior(x, samples, 0.001, 20)
    prior = update_bo_prior(prior, covariances, names, [])
    for i in range(init_num_models):
        prior = update_bo_prior(prior, [], [], i)

    total_time_update = []
    total_time_acquisition = []
    total_time_model_search = []
    not_optimal = True

    for i in range(problem.budget):
        if not_optimal:
            tstart_model_search = time.time()
            boms_models, boms_log_evidence, prior = aboms_wrapper(problem, boms_models, prior, x, y)
            boms_log_evidence = boms_log_evidence
            new_models = boms_models[-problem.eval_budget:]
            new_log_evidence = boms_log_evidence[-problem.eval_budget:]
            new_log_evidence = new_log_evidence * len(y)
            tfinish_model_search = time.time()
            time_model_search = tfinish_model_search - tstart_model_search
            print("Total time for model search: " + str(time_model_search))
            total_time_model_search.append(time_model_search)

            tstart_update = time.time()
            if i > 0:
                models, log_evidence = gp_update(problem, models, x, y)

            models = models + new_models
            log_evidence = log_evidence + new_log_evidence
            models = sorted(models, key=lambda idx: log_evidence[models.index(idx)], reverse=True)
            log_evidence.sort(reverse=True)

            # computing model_posterior
            # exp and model evidence normalization
            model_posterior = np.exp(log_evidence - max(log_evidence))
            model_posterior = model_posterior / sum(model_posterior)

            for j in range(len(model_posterior)):
                if (model_posterior[j] < 0.01) or (j > problem.max_num_models):
                    final_model = j - 1
                    models = models[:final_model]
                    # model_posterior = model_posterior[:final_model]
                    # model_posterior = model_posterior / sum(model_posterior)
                    break
            tfinish_update = time.time()
            time_update = tfinish_update - tstart_update
            print("Total time for model update: " + str(time_update))
            total_time_update.append(time_update)

            tstart_acquisition = time.time()
            chosen_x_star = expected_improvement_limited_optimization(problem, models, x, y)
            y = problem.f.eval(chosen_x_star)
            tfinish_acquisition = time.time()
            # update lists with new observation
            x.append(chosen_x_star)
            y.append(y)
            time_acquisition = tfinish_acquisition - tstart_acquisition
            print("Total time for model acquisition: " + str(time_acquisition))
            total_time_acquisition.append(time_acquisition)
        y_first = min(problem.initial_y)
        y_best = min(y)
        gap = (y_first - y_best) / (y_first - problem.optimum)
        if gap > 0.995 and abs(y_best - problem.optimum) < 0.0001:
            not_optimal = False

    return x_star, y_star, models


def aboms_wrapper(problem, models, prior, x, y):
    covariances_root = problem.covariance_root
    theta = Theta(problem.data_noise)
    covariances = covariance_grammar_started(covariances_root, theta, problem.d)
    for i in range(len(covariances)):
        covariances[i].fixed_hyps = np.full((len(covariances[i].priors), 1), float('inf'))
    if problem.d > 1:
        covariances = mask_kernels(covariances, problem.d)
    problem.theta_models = theta
    problem.n = x.shape[0]
    problem.points = x
    problem.y = y
    problem.covariances = covariances
    best_model, models, prior, log_evidence = aboms(problem, models, prior)
    return models, log_evidence, prior


def aboms(problem, *args):
    # Constants
    total_hyp_samples = problem.total_hyp_samples
    covariances = problem.covariances
    num_models = len(covariances)
    bo_data_noise = 0.001
    base_covs = covariances
    max_hyps = 200
    max_depth = 10
    eval_budget = problem.eval_budget
    exploit_budget = problem.exploit_budget
    explore_budget = problem.explore_budget
    starting_depth = 2
    max_candidates = problem.max_candidates
    top_k = problem.k
    data_subsample_size = 20
    expand_best = False

    do_BOMS = len(args) == 0

    # Initial setup
    models = []
    base_names = [covariance2str(cov.fun) for cov in covariances]
    base_hyps = [len(cov.priors) for cov in covariances]

    train_order = []

    if do_BOMS:
        covariances, names = fully_expand_tree(base_covs, starting_depth, max_candidates)
        num_models = len(covariances)
        samples = sobol_sample(total_hyp_samples, max_hyps)
        prior = create_bo_prior(problem.points, samples, problem.data_noise, data_subsample_size)
        prior = update_bo_prior(prior, covariances, names, [])
    else:
        # ABOMS
        init_models = args[0]
        init_cov = []
        for i in range(len(init_models)):
            temp_cov = Covariance()
            temp_cov.fun = init_models[i].parameters.covariance_function
            temp_cov.priors = init_models[i].parameters.priors.cov
            init_cov.append(temp_cov)
        init_names = [covariance2str(cov.fun) for cov in init_cov]

        models = args[0]
        prior = args[1]

    x = np.array([])
    y = np.array([])
    x_star = np.arange(num_models)
    next_x = 0  # train SE model first

    hyps = []
    t = {'cov': [], 'eval': [], 'bo_eval': [], 'expand': [], 'ei': [], 'total': []}

    best_log_evidence = -np.inf
    best_model_index = 0
    best_scores = -np.inf * np.ones(top_k)
    best_indices = np.zeros(top_k)

    if not do_BOMS:
        init_num_models = len(models)
        x = np.arange(init_num_models)
        x_star = prior['candidates']
        next_x = np.random.choice(prior['candidates'])
        y = np.zeros(init_num_models)
        for i in range(init_num_models):
            y[i] = init_models[i]['log_evidence'] / init_models[i]['number_points']
            t = init_models[i]['parameters']['optimization_time']
            models[i] = init_models[i]
            t['eval'].append(t)

        order = np.argsort(y)[::-1]
        best_log_evidence = y[order[0]]
        best_model_index = order[0]

        best_scores = -np.inf * np.ones(top_k)
        best_indices = np.zeros(top_k)

        for i in range(min(top_k, len(order)) - 1):
            best_scores[i] = y[order[i + 1]]
            best_indices[i] = order[i + 1]

    neighborhoods = [[] for _ in range(top_k)]

    # Begin Search
    for b in range(eval_budget):
        # Train next model.
        model_name = covariance2str(prior['covariances'][next_x].fun)
        new_model = gpr_model_builder(prior['covariances'][next_x], problem.theta_models)
        models.append(new_model)
        models[-1], next_y = gp_update(problem, models[-1], problem.points, problem.y)

        models[-1]['log_evidence'] = next_y
        models[-1]['parameters']['number_points'] = problem.n

        next_y /= problem['n']
        print(
            f'BOMS. Query > Log evidence/n {next_y} Model {covariance2str(models[-1]["parameters"]["covariance_function"])} n = {problem["n"]}')

        num_models = len(models)
        t['eval'].append(models[-1]['parameters']['optimization_time'])

        # Update best models and results
        new_best = False
        if best_log_evidence < next_y:
            best_log_evidence = next_y
            best_model_index = num_models
            new_best = True

        best_scores = np.append(best_scores, next_y)
        best_indices = np.append(best_indices, next_x)
        neighborhoods.append([])
        order = np.argsort(best_scores)[::-1]

        best_scores = best_scores[order[:-1]]
        best_indices = best_indices[order[:-1]]
        neighborhoods = [neighborhoods[i] for i in order[:-1]]
        train_order.append(model_name)

        # Expand candidate pool
        if any(next_x == best_indices):
            neighborhoods[np.where(best_indices == next_x)[0][0]] = expand_covariance(prior['covariances'][next_x],
                                                                                      base_covs, base_names, max_depth)

        new_covs, new_names = select_new_candidates(problem, explore_budget, exploit_budget, prior['names'],
                                                    neighborhoods,
                                                    starting_depth)

        if new_best and expand_best:
            covs_best, names_best = remove_duplicate_candidates(neighborhoods[0], prior['names'] + new_names,
                                                                base_names)
            new_covs.extend(covs_best)
            new_names.extend(names_best)

        # Update datasets and cov names
        x = np.arange(num_models)
        y = np.append(y, next_y)

        if new_covs:
            x_star = np.arange(num_models + 1, prior['n'] + len(new_covs))
        else:
            x_star = np.arange(num_models + 1, prior['n'])

        # Update BO model
        prior = update_bo_prior(prior, new_covs, new_names, next_x)
        best_indices[best_indices == next_x] = num_models

        kernel_kernel = boms_model_builder(prior['cov'], Theta(bo_data_noise), prior['mean'])

        # if not do_BOMS:
        #     kernel_kernel['parameters']['covariance_function'] = [
        #                                                          @ add_noise_covariance,
        #     {'candidates': list(range(len(init_models))), 'noise': noise_old_models,
        #      'cov': kernel_kernel['parameters']['covariance_function']}
        #     ]

        try:
            kernel_kernel['parameters'] = gp_train(x, y, kernel_kernel['parameters'])
        except:
            # removing models that are very similar
            nn = y.shape[0]
            exclude = np.zeros(nn, dtype=bool)
            for i in range(nn):
                for j in range(i + 1, nn):
                    if np.linalg.norm(y[i] - y[j]) < 0.01:
                        exclude[i] = True
            x = x[~exclude]
            y = y[~exclude]
            t['eval'] = np.array(t['eval'])[~exclude]
            kernel_kernel['parameters'] = gp_train(x, y, kernel_kernel['parameters'])

        kernel_kernel_gp = kernel_kernel['parameters']

        # update timing model (OLS)
        num_hyps_x = get_num_hyps(prior['names'][x], base_names, base_hyps).reshape(-1, 1)
        if b < 4:
            theta = np.array([0, .1])
        else:
            x_t = np.hstack([np.ones_like(num_hyps_x), num_hyps_x])
            y_t = np.log(t['eval']).reshape(-1, 1)
            theta = np.linalg.solve(x_t.T @ x_t + 0.01 * np.eye(x_t.shape[1]), x_t.T @ y_t).flatten()

        # Select next model to evaluate
        mu, s2 = gp(kernel_kernel_gp['theta'], 'exact', kernel_kernel_gp['mean_function'], kernel_kernel_gp[
            'covariance_function'], 'Gaussian', x, y, x_star)

        num_hyps = get_num_hyps(prior['names'][x_star], base_names, base_hyps)
        times = np.exp(theta[0] + theta[1] * num_hyps)

        if b < eval_budget:
            next_x, scores = select_best_candidate(np.max(y), x_star, mu, s2, times)
            order = np.argsort(scores)
            for i in order[:5]:
                print(
                    f'ei/s: {scores[i]} model {covariance2str(prior["covariances"][x_star[i]].fun)} ({mu[i]} +- {2 * np.sqrt(s2[i])})')

        # reduce to max # of candidates
        num_candidates = len(order)
        if num_candidates > max_candidates:
            cand_to_remove = x_star[order[:num_candidates - max_candidates]]
            prior = update_bo_prior(prior, [], [], [], cand_to_remove)
            next_x -= np.sum(cand_to_remove < next_x)

        best_model = models[best_model_index]
        return best_model, models, prior, y


def boms_model_builder(covariance_model, hyper_or_noise, mean_model=None):
    # Likelihood specification
    likelihood = GPy.likelihoods.Gaussian(variance=hyper_or_noise.lik_noise_std ** 2)
    # Mean function specification
    if mean_model is not None:
        mean_function = mean_model.fun
    else:
        mean_function = GPy.mappings.constant.Constant(input_dim=1, output_dim=1, value=hyper_or_noise.mean_offset)
    # Covariance function specifications
    cov_function = covariance_model.fun
    # Creating the GP model
    gp_model = GPy.core.GP(X=None, Y=None, kernel=cov_function, likelihood=likelihood,
                           mean_function=mean_function)
    # Priors (using Gaussian priors as an example, similar to GPML toolbox)
    if mean_model is not None:
        gp_model.mean_function.set_prior(mean_model.priors)
    else:
        gp_model.mean_function.set_prior(
            GPy.priors.Gaussian(hyper_or_noise.mean_offset, np.sqrt(hyper_or_noise.mean_var)))
    gp_model.likelihood.variance.set_prior(
        GPy.priors.Gaussian(hyper_or_noise.lik_noise_std, np.sqrt(hyper_or_noise.lik_noise_std_var)))

    for i, prior in enumerate(covariance_model.priors):
        gp_model.kern.parameters[i].set_prior(GPy.priors.Gaussian(prior[0], np.sqrt(prior[1])))

    model = {
        'parameters': gp_model,
        'log_evidence': None,
        # 'update': gp_update,
        # 'entropy': gpr_entropy,
    }
    return model


def boms_hyperparameters(data_noise=0.5):
    param = {}

    param['length_scale_mean'] = np.log(0.1)
    param['length_scale_var'] = 0.5
    param['p_mean'] = np.log(0.1)
    param['p_var'] = 0.5

    param['output_scale_mean'] = np.log(0.4)
    param['output_scale_var'] = 0.5

    param['p_length_scale_mean'] = np.log(2)
    param['p_length_scale_var'] = 0.5

    param['alpha_mean'] = np.log(0.05)
    param['alpha_var'] = 0.5

    param['lik_noise_std'] = np.log(data_noise)
    param['lik_noise_std_var'] = 1

    param['offset_mean'] = 0
    param['offset_var'] = 10 * param['output_scale_var']

    param['mean_offset'] = 0
    param['mean_var'] = 1

    param['model_temperature_mean'] = np.log(25)
    param['model_temperature_var'] = np.log(25)

    param['model_offset_mean'] = 0
    param['model_offset_var'] = np.log(1.5)

    param['model_length_scale_mean'] = np.log(0.5)
    param['model_length_scale_var'] = 1

    param['model_output_scale_mean'] = np.log(0.4)
    param['model_output_scale_var'] = 1

    return param
