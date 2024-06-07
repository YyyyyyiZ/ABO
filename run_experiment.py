import argparse
from test_function import *
from util.problem import *
from gp.gp_model import *
from bo.abo import abo

# Set up the objective function
parser = argparse.ArgumentParser('Run ABO Experiments')
# ----------------try other functions----------------
parser.add_argument('--function_name', type=str, default='branin')
parser.add_argument('--covariances_root', type=str, nargs='+', default=["SE", "RQ"], help='default ["SE", "RQ"]')
parser.add_argument('--num_queries', type=int, default=20, help='number of trials for the experiment')
parser.add_argument('--n_init', type=int, default=5, help='number of initialising random points')
parser.add_argument('--acq', type=str, default='ei', help='choice of the acquisition prediction function.')
parser.add_argument('--exploit_budget', type=int, default=10, help='exploit budget')
parser.add_argument('--explore_budget', type=int, default=5, help='explore budget')
parser.add_argument('--data_noise', type=float, default=0.01, help='data noise')
parser.add_argument('--seed', type=int, default=None, help='**initial** seed setting')

args = parser.parse_args()
options = vars(args)
print(options)

if args.problem == 'branin':
    f = Branin()
elif args.problem == 'levy3':
    f = Levy(dim=3)
elif args.problem == 'griewank5':
    f = Griewank(dim=5)
elif args.problem == 'hartmann6':
    f = Hartmann6()
else:
    raise ValueError('Unrecognised problem type %s' % args.problem)
d = f.dim
lb = f.lb
ub = f.ub
optimum = f.min
theta = Theta(args.data_noise)

problem = Problem(fun=args.problem, lb=lb, ub=ub, d=d, optimum=optimum,
                  covariances_root=args.covariances_root, opt=None, num_queries=args.num_queries,
                  max_num_models=args.max_num_models, prediction_function=args.prediction_function, max_candidates=200,
                  boms_eval_budget=5, exploit_budget=args.exploit_budget, explore_budget=args.explore_budget,
                  total_hyp_samples=100, param_k=3, data_noise=args.data_noise)

x_init = np.random.uniform(lb, ub, (args.n_init, d))
y_init = f.eval(x_init)
problem.initial_x = x_init
problem.initial_y = y_init

initial_models = get_initial_models(problem, d, args.data_noise)

# # Run Active GP Learning
# if args.covariances_root is not None:
#     covariances_root = covariance_grammar_started(args.covariances_root, theta, d)
#     initial_models = gpr_model_builder(problem, covariances_root[0], theta)

models = initial_models

x_star, y_star, models = abo(problem, models, args.acq)
