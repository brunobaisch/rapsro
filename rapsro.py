import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from scipy import stats
import pickle
import json
from numpy.random import RandomState
import argparse
import multiprocessing as mp
from scipy.optimize import minimize
import itertools
import pandas as pd
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
my_path = 'C:/Users/bruno/OneDrive/Área de Trabalho/MSc Machine Learning-DESKTOP-N01V9DD/Project/code/results/20210805-145428_50_uniform_True/'

np.random.seed(0)

parser = argparse.ArgumentParser(description='Arguments for RAPSRO experiments')

optimizer = 'scipy'
parser.add_argument('--game', type=str, default='gos')
parser.add_argument('--dim_list', type=list, default=[50])
parser.add_argument('--nb_iters', type=int, default=[40, 40])
parser.add_argument('--num_experiments', type=int, default=8)
parser.add_argument('--num_threads', type=int, default=8)
parser.add_argument('--mixed', type=bool, default=True)
parser.add_argument('--iters_nfg', type=int, default=1000)
parser.add_argument('--dpp_gamma', type=int, default=20)
parser.add_argument('--gamma_list', type=list, default=[20, 6, -6, -20])
#parser.add_argument('--gamma_list', type=list, default=[20, 6])
parser.add_argument('--calc_ef', type=bool, default=True)
parser.add_argument('--load_results', type=bool, default=False)
parser.add_argument('--calc_risk_ne', type=bool, default=True)
parser.add_argument('--method', type=str, default='gamma')
parser.add_argument('--er', type=float, default=-0.001)
args = parser.parse_args()

# metrics that will be plotted in results charts
metrics = ['exp', 'cardinality', 'er_risk', 'er_ne', 'er_unif',
           'var_risk', 'var_ne', 'var_unif', 'var_exp', 'entropy', 'runtime', 'strat']
LR = 0.5
TH = 0.03
gap = 0.025
er_range = np.arange(-1, 1.05, gap)

expected_card = []
sizes = []

root_path='C:/Users/bruno/OneDrive/Área de Trabalho/MSc Machine Learning-DESKTOP-N01V9DD/Project/code'
data_path = 'C:/Users/bruno/OneDrive/Área de Trabalho/MSc Machine Learning-DESKTOP-N01V9DD/Project/Images'

def run_dummy_exp(gamma_list, PATH_RESULTS):
    # payoffs = np.array([[1, -101],
    #                     [-1, 100]])
    payoffs = np.array([[1, -3],
                        [-1, 2]])
    strat = 0.5*np.ones(2)
    results = {}
    results[0] = get_br_to_strat(strat, payoffs=payoffs)
    for gamma in gamma_list:
        results[gamma] = get_br_to_strat_risk(strat, payoffs=payoffs, gamma=gamma)

    # plot results
    labels = [str(gamma) for gamma in results.keys()]
    heights = [result[1] for result in results.values()]
    fig_handle = plt.figure()
    plt.bar(x=labels, height=heights)
    plt.title('Weights in action 2')
    plt.xlabel('gamma')
    plt.savefig(os.path.join(PATH_RESULTS, 'dummy_example.pdf'))

    return results

def clean_range(r):
    result = [r[i] if r[i] != r[i + 1] else np.nan for i in range(r.shape[0] - 1)]  # remove repeated values
    return result

def plot_ef(results, params, PATH_RESULTS):
    var_dict = {}
    var_dict['unif'] = clean_range(np.nanmean(results['var_range'], axis=0))
    var_dict['ne'] = clean_range(np.nanmean(results['var_range_ne'], axis=0))
    var_dict['risk'] = clean_range(np.nanmean(results['var_range_risk'], axis=0))

    scatter_er = {}
    scatter_var = {}
    #opponents = ['unif', 'ne', 'risk']
    opponents = ['ne']
    # get scatter plots for all the relevant algos
    for opponent in opponents:
        scatter_er[opponent] = {}
        scatter_var[opponent] = {}
        if params['psro_risk']:
            for gamma in params['gamma_list']:
                if gamma > 0:
                    er_array = np.nanmean(results['psro_risk_g' + str(gamma) + '_er_' + opponent], axis=0).squeeze()
                    var_array = np.nanmean(results['psro_risk_g' + str(gamma) + '_var_' + opponent], axis=0).squeeze()
                    my_key = 'rapsro_g' + str(gamma)
                    scatter_er[opponent][my_key] = er_array[-1]
                    s = np.where(er_range > er_array[-1])[0][0]-1
                    slope = (var_dict[opponent][s+1] - var_dict[opponent][s])/gap
                    var_limit = var_dict[opponent][s] + slope*(er_array[-1]-er_range[s])
                    scatter_var[opponent][my_key] = np.maximum(var_array[-1], var_limit)

        if params['psro']:
            er_array = np.nanmean(results['psro_er_' + opponent], axis=0).squeeze()
            var_array = np.nanmean(results['psro_var_' + opponent], axis=0).squeeze()
            scatter_er[opponent]['psro'] = er_array[-1]
            s = np.where(er_range > er_array[-1])[0][0] - 1
            if s < len(var_dict[opponent]) - 1:
                slope = (var_dict[opponent][s + 1] - var_dict[opponent][s]) / gap
            else:
                slope = (var_dict[opponent][s] - var_dict[opponent][s-1]) / gap
            var_limit = var_dict[opponent][s] + slope * (er_array[-1] - er_range[s])
            scatter_var[opponent]['psro'] = np.nanmax([var_array[-1], var_limit])

        if params['dpp_psro']:
            er_array = np.nanmean(results['dpp_psro_er_' + opponent], axis=0).squeeze()
            var_array = np.nanmean(results['dpp_psro_var_' + opponent], axis=0).squeeze()
            scatter_er[opponent]['dpp'] = er_array[-1]
            s = np.where(er_range > er_array[-1])[0][0] - 1
            if s < len(var_dict[opponent]) - 1:
                slope = (var_dict[opponent][s + 1] - var_dict[opponent][s]) / gap
            else:
                slope = (var_dict[opponent][s] - var_dict[opponent][s-1]) / gap
            var_limit = var_dict[opponent][s] + slope * (er_array[-1] - er_range[s])
            scatter_var[opponent]['dpp'] = np.maximum(var_array[-1], var_limit)

        if params['dpp_risk']:
            er_array = np.nanmean(results['dpp_risk_er_' + opponent], axis=0).squeeze()
            var_array = np.nanmean(results['dpp_risk_var_' + opponent], axis=0).squeeze()
            scatter_er[opponent]['dpp_risk'] = er_array[-1]
            s = np.where(er_range > er_array[-1])[0][0] - 1
            if s < len(var_dict[opponent]) - 1:
                slope = (var_dict[opponent][s + 1] - var_dict[opponent][s]) / gap
            else:
                slope = (var_dict[opponent][s] - var_dict[opponent][s-1]) / gap
            var_limit = var_dict[opponent][s] + slope * (er_array[-1] - er_range[s])
            scatter_var[opponent]['dpp_risk'] = np.maximum(var_array[-1], var_limit)

    # get er and var for standard best response
    #er_ne = np.mean(results['er_br'], axis=0)
    #var_ne = np.mean(results['var_br'], axis=0)
    er_ne = np.mean(results['er_ne'], axis=0)
    var_ne = np.mean(results['var_ne'], axis=0)

    #plot frontier for ne
    fig, ax = plt.subplots()
    opponent = 'ne'
    zero_pos = np.where(er_range > -0.01)[0][0]
    #ax.plot(var_dict[opponent], er_range[:-1], label='efficient frontier')
    ax.plot(var_dict[opponent][:zero_pos+1], er_range[:zero_pos+1], label='efficient frontier')
    ax.scatter(var_ne, er_ne, marker='x', label='ne strat', c='r')
    x_val = list(scatter_var[opponent].values())
    y_val = list(scatter_er[opponent].values())
    ax.scatter(x_val, y_val, marker='D', c='g')
    for j, txt in enumerate(scatter_er[opponent].keys()):
        #if txt != 'rapsro_g6':
            ax.annotate(txt, (x_val[j], y_val[j]))
    ax.legend(loc='lower right')
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
    ax.set(ylabel='expected return', xlabel='variance', title='EF vs NE')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_RESULTS, 'efficient_frontier_ne.pdf'))

def plot_runtime(results, params, PATH_RESULTS):
    runtime_results = {}
    if params['psro_risk']:
        for gamma in args.gamma_list:
            if gamma > 0:
                my_key = 'psro_risk_g' + str(gamma)
                runtime_results[my_key] = np.mean(results[my_key + '_runtime']) / 60
    if params['psro']:
        runtime_results['psro'] = np.mean(results['psro_runtime']) / 60
    if params['dpp_psro']:
        runtime_results['dpp_psro'] = np.mean(results['dpp_psro_runtime']) / 60

    fig, ax = plt.subplots()
    ax.bar(x=list(runtime_results.keys()), height=runtime_results.values())
    ax.set(title='Average runtime', ylabel='avg runtime (min)')
    plt.savefig(os.path.join(PATH_RESULTS, 'runtime.pdf'))

def plot_dist(results, params, PATH_RESULTS):
    strat_ne = np.array(results['strat_ne'])
    runtime_results = {}
    labels = []
    if params['psro_risk']:
        for gamma in args.gamma_list:
            if gamma > 0:
                my_key = 'psro_risk_g' + str(gamma) + '_strat'
                strat = np.array(results[my_key]).squeeze()
                runtime_results[my_key] = np.mean(calc_euc_dist(strat, strat_ne))
                labels.append('rapsro_g' + str(gamma))
    if params['psro']:
        my_key = 'psro_strat'
        strat = np.array(results[my_key]).squeeze()
        runtime_results[my_key] = np.mean(calc_euc_dist(strat, strat_ne))
        labels.append('psro')
    if params['dpp_psro']:
        my_key = 'dpp_psro_strat'
        strat = np.array(results[my_key]).squeeze()
        runtime_results[my_key] = np.mean(calc_euc_dist(strat, strat_ne))
        labels.append('dpp')

    fig, ax = plt.subplots()
    ax.bar(x=labels, height=runtime_results.values())
    ax.set(title='Eucledian Distance to NE', ylabel='eucledian distance to NE')
    plt.savefig(os.path.join(PATH_RESULTS, 'dist_to_ne.pdf'))

# Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

def get_br_to_strat_risk(strat, payoffs=None, verbose=False, gamma=1, mixed=True, alpha=1, method='gamma', er=-0.001):
    """assume the row player is playing strat, and we as the column player
    are trying to best respond"""

    row_weighted_payouts = strat @ payoffs  # expected payoff to row player, for each action of the column player. the column player wants this to be as little as possible
    dim = strat.shape[0]
    #construct covariance matrix
    if dim > 1:
        cov = np.cov(payoffs.T, aweights=strat)
    else:
        cov = np.array([0]).reshape(1, 1)

    #inputs into optimisation

    if mixed and gamma > 0 and method == 'gamma':
        if optimizer == 'scipy':
            #start = time.time()
            objective = lambda w: alpha * w @ row_weighted_payouts + gamma * w @ cov @ w
            jac = lambda w: alpha*row_weighted_payouts + 2*gamma*cov@w
            hess = lambda w: 2*gamma*cov
            cons = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
            bnds = [(0, 1) for _ in range(strat.shape[0])]
            x0 = np.ones(strat.shape[0]) / strat.shape[0]
            res = minimize(objective, x0, jac=jac, hess=hess, bounds=bnds, constraints=cons)
            br = res.x  # store results
            #end = time.time()
        else:
            #start = time.time()
            w = cp.Variable(dim)
            risk = cp.quad_form(w, psd_wrap(cov))
            objective = cp.Minimize(alpha * w @ row_weighted_payouts + gamma * risk)
            constraints = [0 <= w, w <= 1, cp.sum(w) == 1]
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            br = w.value
            #end = time.time()
            #print('type 2: {0}'.format(end - start))
    elif method == 'min_var':
        objective = lambda w: w @ cov @ w
        jac = lambda w: 2 * cov @ w
        hess = lambda w: 2 * cov
        cons1 = {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)}
        cons2 = {'type': 'ineq', 'fun': lambda w: w @ payoffs @ strat - er}
        cons = [cons1, cons2]
        bnds = [(0, 1) for _ in range(strat.shape[0])]
        x0 = np.ones(strat.shape[0]) / strat.shape[0]
        res = minimize(objective, x0, jac=jac, hess=hess, bounds=bnds, constraints=cons)
        br = res.x  # store results
    else:  # consider only pure strategies as best response
        objective = lambda w: alpha * w @ row_weighted_payouts + gamma * w @ cov @ w
        n = strat.shape[0]
        br_matrix = np.eye(n)
        values = [objective(br_matrix[:, i]) for i in range(n)]
        br = np.zeros(strat.shape)
        br[np.argmin(values)] = 1

    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

def get_min_var(strat, payoffs=None, expected_return=None, init_arr=None):
    """given a certain strategy for the opponent, find the response that minimizes variance subject
    to a expected return constrain"""

    cov = np.cov(payoffs.T, aweights=strat)  # construct covariance matrix
    objective = lambda w: w@cov@w
    jac = lambda w: 2*cov@w
    hess = lambda w: 2*cov
    cons1 = {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)}
    cons2 = {'type': 'ineq', 'fun': lambda w: w @ payoffs @ strat - expected_return}
    cons = [cons1, cons2]
    bnds = [(0, 1) for _ in range(strat.shape[0])]

    ntries = init_arr.shape[1]
    var_list = []
    for i in range(ntries):
        x0 = init_arr[:, i].squeeze()
        res = minimize(objective, x0, jac=jac, hess=hess, bounds=bnds, constraints=cons)
        br = res.x  # store results
        if res.success:
            var_list.append(br@cov@br)
        else:
            var_list.append(np.nan)

    var = np.nanmin(var_list)
    return var

def get_min_var_cp(strat, payoffs=None, expected_return=None, init_arr=None):
    #using cvxpy
    dim = strat.shape[0]
    cov = np.cov(payoffs.T, aweights=strat)  # construct covariance matrix
    w = cp.Variable(dim)
    er = w@payoffs@strat
    objective = cp.Minimize(cp.quad_form(w, psd_wrap(cov)))
    constraints = [0 <= w, w <= 1, cp.sum(w) == 1, er >= expected_return]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return result

# Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=None, verbose=False, gamma=10, risk=False, mixed=True, method='gamma', er=-0.001):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        if risk:
            br = get_br_to_strat_risk(average, payoffs=payoffs, gamma=gamma, mixed=mixed, method=method, er=er)
        else:
            br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))

    pop_diff = calc_euc_dist(averages, pop)
    return averages, exps, pop_diff

# Solve exploitability of a nash equilibrium over a fixed population
def get_exploitability(pop, payoffs, iters=500, risk=False, gamma=10, mixed=True, method='gamma', er=-0.001):
    # calculate exploitability
    emp_game_matrix = pop @ payoffs @ pop.T

    # compute your strategy
    averages, _, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters, risk=risk, gamma=gamma, method=method, er=er)  # compute ne
    strat = averages[-1] @ pop  # this gives you the player final strategy at the action level

    # compute exploitability
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp1 = strat @ payoffs @ test_br.T  # payoff to row player if they play strat and column player best responds
    exp2 = test_br @ payoffs @ strat  # payoff to row player if column player plays strat and row player best responds
    exploitability = exp2 - exp1

    # compute variance exploitability
    cov = np.cov(payoffs.T, aweights=strat)
    var_br = get_br_to_strat_risk(strat, payoffs=payoffs, gamma=-1, alpha=0)
    var_exp = var_br @ cov @ var_br
    # print('var_exp is ', str(var_exp))

    # calculate expected return and variance
    er_risk = strat @ payoffs @ strat_risk
    er_ne = strat @ payoffs @ strat_ne
    er_unif = strat @ payoffs @ strat_unif

    # calculate variance
    var_risk = strat_risk @ cov @ strat_risk
    var_ne = strat_ne @ cov @ strat_ne
    var_unif = strat_unif @ cov @ strat_unif

    # calculate entropy
    entropy = stats.entropy(strat)

    return {
            'exp': exploitability,
            'var_exp': var_exp,
            'er_risk': er_risk,
            'er_ne': er_ne,
            'er_unif': er_unif,
            'var_risk': var_risk,
            'var_ne': var_ne,
            'var_unif': var_unif,
            'entropy': entropy,
            'strat': list(strat)
            }

def calc_euc_dist(arr1, arr2):
    diff = np.sum((arr1-arr2)**2, axis=1)
    if arr1.ndim > 1:
        diff = diff[1:]
    return diff

def dpp_opt(w, pop, payoffs, meta_nash, k, gamma, lr):
    aggregated_enemy = meta_nash @ pop[:k]
    pop_k = lr * w + (1 - lr) * pop[k]  #mixing the new proposal with the latest agent in the population
    pop_tmp = np.vstack((pop[:k], pop_k))
    w_mix = np.minimum(k, 9)
    meta_nash_new = np.append(w_mix/(w_mix+1)*meta_nash, 1/(w_mix+1))
    new_strat = meta_nash_new @ pop_tmp

    row_weighted_payouts = aggregated_enemy @ payoffs
    cov = np.cov(payoffs.T, aweights=aggregated_enemy)
    objective = new_strat @ row_weighted_payouts + gamma * new_strat @ cov @ new_strat
    return objective

def joint_loss(pop, payoffs, meta_nash, k, lambda_weight, lr, risk=False, gamma=10, mixed=False, method='gamma', er=-0.001):
    dim = payoffs.shape[0]
    br = np.zeros((dim,))
    values = []
    cards = []
    aggregated_enemy = meta_nash @ pop[:k]
    for i in range(dim):
        #looping through making the best response put all probability in a single action
        #you then replace the last item in the population by the one that plays this single action
        br_tmp = np.zeros((dim, ))
        br_tmp[i] = 1.

        if risk: #compute risk aware value
            cov = np.cov(payoffs.T, aweights=aggregated_enemy)
            value = br_tmp @ payoffs @ aggregated_enemy.T - gamma*br_tmp @ cov @ br_tmp
        else:
            value = br_tmp @ payoffs @ aggregated_enemy.T  #temporary estimate of your policy value

        pop_k = lr * br_tmp + (1 - lr) * pop[k]
        pop_tmp = np.vstack((pop[:k], pop_k))
        M = pop_tmp @ payoffs @ pop_tmp.T
        #metanash_tmp, _ = fictitious_play(payoffs=M, iters=500)
        # L = np.diag(metanash_tmp[-1]) @ M @ M.T @ np.diag(metanash_tmp[-1])
        #you work out what the L and l_card are based on the new M
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))

        cards.append(l_card)
        values.append(value)

    # with probability lambda, your best response is the one that gives you the highest value
    # with prob 1-lambda, it's the one with highest card value
    if np.random.randn() < lambda_weight:
        if risk and mixed:
            br = get_br_to_strat_risk(aggregated_enemy, payoffs=payoffs, gamma=gamma, method=method, er=er)
        else:
            br[np.argmax(values)] = 1
    else:
        if risk and mixed:
            cons = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
            bnds = [(0, 1) for _ in range(dim)]
            x0 = np.ones(dim) / dim
            res = minimize(dpp_opt, x0, args=(pop, payoffs, meta_nash, k, gamma, lr), bounds=bnds, constraints=cons)
            br = res.x  # store results
        else:
            br[np.argmax(cards)] = 1
    return br

def psro_steps_risk(iters=5, payoffs=None, verbose=False, seed=0,
                        num_learners=4, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False,
                        gamma=10, mixed=True, method='gamma', er=-0.001):
    start_psro = time.time()
    risk = True
    dim = payoffs.shape[0]
    lambda_weight = 0  # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
    r = np.random.RandomState(seed)
    np.random.seed(seed)
    pop = np.random.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    result = get_exploitability(pop, payoffs, iters=500, risk=risk, gamma=gamma)
    results = {key: [value] for key,value in result.items()}

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    #l_cards = [l_card]
    results['cardinality'] = [l_card]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    start = time.time()
    for i in range(iters):
        if i % 5 == 0:
            end = time.time()
            print('iteration: ', i, ' exp full: ', results['exp'][-1], 'runtime: ', end-start)
            print('size of pop: ', pop.shape[0])
            start = time.time()

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            if emp_game_matrix.shape[0] > 1:
                meta_nash, _, _ = fictitious_play(payoffs=emp_game_matrix, iters=500, risk=risk, gamma=gamma, method=method, er=er)
                # meta_nash, _, _ = fictitious_play(payoffs=emp_game_matrix, iters=500, risk=False)
            else:
                meta_nash = np.array([[1.]])
            #meta_nash, _, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000, risk=False)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash

            # with standard psro, you adjust your policy so it is a best response to the population strategy
            # with diverse psro, you use the joint loss to include something that accounts for diversity
            if loss_func == 'br':
                # standard PSRO
                br = get_br_to_strat_risk(population_strategy, payoffs=payoffs, gamma=gamma, mixed=mixed, method=method, er=er)
            else:
                # Diverse PSRO
                br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr, risk=True, gamma=gamma, mixed=mixed, method=method, er=er)

            # Update the mixed strategy towards the pure strategy which is returned as the best response to the
            # nash equilibrium that is being trained against.
            pop[k] = lr * br + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # calculate exploitability for meta Nash of whole population
        result = get_exploitability(pop, payoffs, iters=500, risk=risk, gamma=gamma)
        for key,value in result.items():
            results[key].append(value)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        #l_cards.append(l_card)
        results['cardinality'].append(l_card)

    results['strat'] = results['strat'][-1]
    end_psro = time.time()
    results['runtime'] = end_psro - start_psro
    return pop, results

def psro_steps(iters=5, payoffs=None, verbose=False, seed=0,
                        num_learners=4, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    start_psro = time.time()
    dim = payoffs.shape[0]

    np.random.seed(seed)
    pop = np.random.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    result = get_exploitability(pop, payoffs, iters=500, risk=False)
    results = {key: [value] for key, value in result.items()}

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    #l_cards = [l_card]
    results['cardinality'] = [l_card]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    for i in range(iters):
        # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
        lambda_weight = 0.85
        if i % 5 == 0:
            print('iteration: ', i, ' exp full: ', results['exp'][-1])
            print('size of pop: ', pop.shape[0])

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _, _ = fictitious_play(payoffs=emp_game_matrix, iters=500)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash

            #with standard psro, you adjust your policy so it is a best response to the population strategy
            #with diverse psro, you use the joint loss to include something that accounts for diversity
            if loss_func == 'br':
                # standard PSRO
                br = get_br_to_strat(population_strategy, payoffs=payoffs)
            else:
                # Diverse PSRO
                br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                br_orig = get_br_to_strat(population_strategy, payoffs=payoffs)

            # Update the mixed strategy towards the pure strategy which is returned as the best response to the
            # nash equilibrium that is being trained against.
            pop[k] = lr * br + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # calculate exploitability for meta Nash of whole population
        result = get_exploitability(pop, payoffs, iters=500, risk=False)
        for key,value in result.items():
            results[key].append(value)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        #l_cards.append(l_card)
        results['cardinality'].append(l_card)

    results['strat'] = results['strat'][-1]
    end_psro = time.time()
    results['runtime'] = end_psro - start_psro
    return pop, results

# Define the self-play algorithm
def self_play_steps(iters=10, payoffs=None, verbose=False, improvement_pct_threshold=.03, lr=.2, seed=0):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (2, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    result = get_exploitability(pop, payoffs, iters=500, risk=False)
    results = {key:[value] for key,value in result.items()}
    performances = [.01]

    M = pop @ payoffs @ pop.T
    L = M@M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    #l_cards = [l_card]
    results['cardinality'] = [l_card]

    for i in range(iters):
        if i % 10 == 0:
            print('iteration: ', i, 'exploitability: ', results['exp'][-1])
        br = get_br_to_strat(pop[-2], payoffs=payoffs)
        pop[-1] = lr * br + (1 - lr) * pop[-1]
        performance = pop[-1] @ payoffs @ pop[-2].T + 1
        performances.append(performance)
        if performance / performances[-2] - 1 < improvement_pct_threshold:
            learner = np.random.uniform(0, 1, (1, dim))
            learner = learner / learner.sum(axis=1)[:, None]
            pop = np.vstack((pop, learner))
        result = get_exploitability(pop, payoffs, iters=500, risk=False)
        for key,value in result.items():
            results[key].append(value)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        #l_cards.append(l_card)
        results['cardinality'].append(l_card)

    return pop, results

# Define the PSRO rectified nash algorithm
def psro_rectified_steps(iters=10, payoffs=None, verbose=False, eps=1e-2, seed=0,
                         num_start_strats=1, num_pseudo_learners=4, lr=0.3, threshold=0.001):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (num_start_strats, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    result = get_exploitability(pop, payoffs, iters=500, risk=False)
    results = {key: [value] for key, value in result.items()}
    counter = 0

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    #l_cards = [l_card]
    results['cardinality'] = [l_card]

    while counter < iters * num_pseudo_learners:
        if counter % (5 * num_pseudo_learners) == 0:
            print('iteration: ', int(counter / num_pseudo_learners), ' exp: ', results['exp'][-1])
            print('size of population: ', pop.shape[0])

        new_pop = np.copy(pop)
        emp_game_matrix = pop @ payoffs @ pop.T
        averages, _, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(pop.shape[0]):
            if counter > iters * num_pseudo_learners:
                return pop, exps, l_cards
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                new_pop = np.vstack((new_pop, learner))
                idx = new_pop.shape[0] - 1

                current_performance = 0.02
                last_performance = 0.01
                while current_performance / last_performance - 1 > threshold:
                    counter += 1
                    mask = emp_game_matrix[j, :]
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    weights = np.multiply(mask, averages[-1])
                    weights /= weights.sum()
                    strat = weights @ pop
                    br = get_br_to_strat(strat, payoffs=payoffs)
                    new_pop[idx] = lr * br + (1 - lr) * new_pop[idx]
                    last_performance = current_performance
                    current_performance = new_pop[idx] @ payoffs @ strat + 1

                    if counter % num_pseudo_learners == 0:
                        # count this as an 'iteration'

                        # exploitability
                        result = get_exploitability(pop, payoffs, iters=500, risk=False)
                        for key, value in result.items():
                            results[key].append(value)

                        M = pop @ payoffs @ pop.T
                        L = M @ M.T
                        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
                        #l_cards.append(l_card)
                        results['cardinality'].append(l_card)

        pop = np.copy(new_pop)

    return pop, results

def run_experiment(param_seed):
    global strat_unif, strat_ne, strat_risk

    params, seed = param_seed
    iters = params['iters']
    num_threads = params['num_threads']
    dim = params['dim']
    lr = params['lr']
    thresh = params['thresh']

    psro_risk = params['psro_risk']
    psro = params['psro']
    pipeline_psro = params['pipeline_psro']
    dpp_psro = params['dpp_psro']
    dpp_risk = params['dpp_risk']
    rectified = params['rectified']
    self_play = params['self_play']
    gamma_list = params['gamma_list']
    dpp_gamma = params['dpp_gamma']
    sampling = params['sampling']
    mixed = params['mixed']
    iters_nfg = params['iters_nfg']
    method = params['method']
    er = params['er']
    game = params['game']

    print('Experiment: ', seed + 1)
    np.random.seed(seed)
    dof = 5

    #initialize results dict
    names = ['psro_risk_g'+str(gamma) for gamma in gamma_list]
    names.extend(['psro', 'pipeline_psro', 'dpp_psro', 'dpp_risk', 'rectified', 'self_play'])
    comb = itertools.product(names, metrics)
    results_labels = [label[0]+'_'+label[1] for label in comb]
    results = {label: [] for label in results_labels}

    #get payoff matrix
    if args.load_results:
        payoffs = params['old_results']['payoffs'][seed]
    else:
        if params['game'] == 'gos':
            if sampling == 'uniform':
                W = np.random.randn(dim, dim)
                S = np.random.randn(dim, 1)
            elif sampling == 't':
                W = np.random.standard_t(dof, size=(dim, dim))
                S = np.random.standard_t(dof, size=(dim, 1))
            else:
                W = np.random.normal(0, np.sqrt(dof/(dof-2)), size=(dim,dim))
                S = np.random.normal(0, np.sqrt(dof/(dof-2)), size=(dim,1))
            payoffs = 0.5 * (W - W.T) + S - S.T

        elif game == 'rsnfg':
            if sampling == 'uniform':
                payoffs = np.triu(np.random.uniform(-1, 1, size=(dim, dim)), k=1)
            elif sampling == 't':
                payoffs = np.triu(np.random.standard_t(dof, size=(dim, dim)), k=1)
            else:
                payoffs = np.triu(np.random.normal(0, np.sqrt(dof/(dof-2)), size=(dim,dim)), k=1)
            payoffs -= payoffs.T
        elif game == 'elo':
            if sampling == 'uniform':
                S = np.random.uniform(0, 2000, size=(dim,1))
            elif sampling == 't':
                S = np.random.standard_t(dof, size=(dim, 1))
                S = 2000*S/np.abs(S).max()
            else:
                S = np.random.normal(0, np.sqrt(dof/(dof-2)), size=(dim, 1))
                S = 2000 * S / np.abs(S).max()
            sd = np.abs(S)/2000
            noise = np.random.normal(0, sd, size=(dim, dim))
            payoffs = (1 + np.exp(-(S - S.T)/400))**(-1) + noise
            payoffs -= payoffs.T
        elif game == 'disc':
            S = np.random.uniform(-1, 1, size=(4*dim, 2))
            idx = S[:, 1]**2 <= 1 - S[:, 0]**2
            S = S[idx, :][:dim]
            A = np.array([[0, -1],
                          [1, 0]])
            payoffs = S @ A @ S.T

        payoffs /= np.abs(payoffs).max() # normalise payoff matrix
    results['payoffs'] = payoffs

    # calculate equilibria at nfg level
    print('Finding NFG equilibria')
    strat_unif = np.ones(dim)/dim
    results['strat_unif'] = strat_unif

    # ne
    if args.load_results:
        ne_ts = params['old_results']['ne_ts'][seed]
        strat_ne = params['old_results']['strat_ne'][seed]
        br_diff = params['old_results']['ne_diff_ts'][seed]
    else:
        averages, _, br_diff = fictitious_play(payoffs=payoffs, iters=1000, risk=False)
        strat_ne = averages[-1]
        ne_ts = calc_euc_dist(averages[1:], averages[:-1])
    results['ne_ts'] = ne_ts
    results['ne_diff_ts'] = br_diff
    results['strat_ne'] = strat_ne
    results['er_ne'] = strat_ne @ payoffs @ strat_ne
    cov = np.cov(payoffs.T, aweights=strat_ne)
    results['var_ne'] = strat_ne @ cov @ strat_ne

    #risk ne
    for gamma in [5, 20, 50]:
        name = 'risk_g' + str(gamma)
        if args.load_results:
            risk_ts = params['old_results'][name+'_ts'][seed]
            strat_risk = params['old_results']['strat_'+name][seed]
            br_diff = params['old_results'][name+'_diff_ts'][seed]
        else:
            if args.calc_risk_ne:  # only compute the actual rane if the argument requires it
                #start = time.time()
                averages, _, br_diff = fictitious_play(payoffs=payoffs, iters=iters_nfg, risk=True, gamma=gamma)
                #end = time.time()
               # print('runtime is {0}'.format(end-start))
                risk_ts = calc_euc_dist(averages[1:], averages[:-1])
                strat_risk = averages[-1]
            else:
                risk_ts = ne_ts
                strat_risk = strat_ne

        # store results in dictionary
        results[name+'_ts'] = risk_ts
        results[name+'_diff_ts'] = br_diff
        results['strat_'+name] = strat_risk
    strat_risk = results['strat_risk_g20']  # pick the correct final strat risk

    # compute matrix summary stats
    results['mat_mean'] = np.mean(payoffs)
    results['mat_cov'] = np.cov(payoffs.T)
    results['mat_skew'] = stats.skew(payoffs, axis=None)
    results['mat_kurt'] = stats.kurtosis(payoffs, axis=None)
    print('matrix kurtosis is {0}'.format(results['mat_kurt']))

    # get efficient frontier
    if args.calc_ef:
        ntries = 5
        init_arr = np.random.rand(payoffs.shape[0], ntries)
        init_arr /= np.sum(init_arr, axis=0)
        init_arr[:, 0] = 0

        #dont delete this part
        if args.load_results:
            var_range = params['old_results']['var_range'][seed]
            var_range_ne = params['old_results']['var_range_ne'][seed]
            var_range_risk = params['old_results']['var_range_risk'][seed]
        else:
            #var_range = [get_min_var(strat_unif, payoffs=payoffs, expected_return=exp_ret, init_arr=init_arr)
            #             for exp_ret in er_range]
            if optimizer == 'scipy':
                var_range_ne = [get_min_var(strat_ne, payoffs=payoffs, expected_return=exp_ret, init_arr=init_arr)
                                for exp_ret in er_range]
            else:
                var_range_ne = [get_min_var_cp(strat_ne, payoffs=payoffs, expected_return=exp_ret, init_arr=init_arr)
                                for exp_ret in er_range]
            #var_range_risk = [get_min_var(strat_risk, payoffs=payoffs, expected_return=exp_ret, init_arr=init_arr)
            #                  for exp_ret in er_range]
        #results['var_range'] = var_range
        results['var_range_ne'] = var_range_ne
        #results['var_range_risk'] = var_range_risk
        results['var_range'] = var_range_ne
        results['var_range_risk'] = var_range_ne

        # get best responses
        strats = [strat_unif, strat_ne, strat_risk]
        ers = []
        vars = []
        for strat in strats:
            br = get_br_to_strat(strat, payoffs)
            ers.append(br@payoffs@strat)
            cov = np.cov(payoffs.T, aweights=strat)
            vars.append(br@cov@br)
        results['er_br'] = ers
        results['var_br'] = vars

    if psro_risk:
        for gamma in gamma_list:
            print('------PSRO Risk, gamma=', str(gamma), '------')
            pop, result = psro_steps_risk(iters=iters[0], num_learners=1, seed=seed+1,
                                                                  improvement_pct_threshold=thresh, lr=lr,
                                                                  payoffs=payoffs, loss_func='br',
                                                                  gamma=gamma, mixed=mixed,
                                                                  method=method, er=er)

            name = 'psro_risk_g'+str(gamma)
            for key,value in result.items():
                results[name+'_'+key].append(value)

    if psro:
        print('------PSRO------')
        pop, result = psro_steps(iters=iters[1], num_learners=1, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='br')
        name = 'psro'
        for key, value in result.items():
            results[name + '_' + key].append(value)

    if pipeline_psro:
        print('------Pipeline PSRO------')
        pop, result = psro_steps(iters=iters[1], num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='br')
        name = 'pipeline_psro'
        for key, value in result.items():
            results[name + '_' + key].append(value)

    if dpp_psro:
        print('------DPP------')
        pop, result = psro_steps(iters=iters[1], num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='dpp')
        name = 'dpp_psro'
        for key, value in result.items():
            results[name + '_' + key].append(value)

    if dpp_risk:
        print('------DPP Risk------')
        pop, result = psro_steps_risk(iters=iters[1], num_learners=1, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='dpp',
                                      gamma=dpp_gamma)
        name = 'dpp_risk'
        for key, value in result.items():
            results[name + '_' + key].append(value)

    if rectified:
        print('------Rectified------')
        pop, result = psro_rectified_steps(iters=iters[1], num_pseudo_learners=num_threads, payoffs=payoffs, seed=seed+1,
                                         lr=lr, threshold=thresh)
        name = 'rectified'
        for key, value in result.items():
            results[name + '_' + key].append(value)

    if self_play:
        print('------Self-play------')
        pop, result = self_play_steps(iters=iters[1], payoffs=payoffs, improvement_pct_threshold=thresh, lr=lr, seed=seed+1)
        name = 'self_play'
        for key, value in result.items():
            results[name + '_' + key].append(value)

    return results

def run_experiments(num_experiments=1, iters=40, num_threads=20, dim=60, lr=0.6, thresh=0.001, logscale=True,
                    iters_nfg=1000,
                    psro_risk=False,
                    psro=False,
                    pipeline_psro=False,
                    rectified=False,
                    self_play=False,
                    dpp_psro=False,
                    dpp_risk=False,
                    gamma_list=[1, 10, 100],
                    dpp_gamma=10,
                    sampling='uniform',
                    mixed=True,
                    metrics=['exp'],
                    method='gamma',
                    er=-0.001,
                    game='gos'
                    ):

    params = {
        'num_experiments': num_experiments,
        'iters': iters,
        'num_threads': num_threads,
        'dim': dim,
        'lr': lr,
        'thresh': thresh,
        'iters_nfg': iters_nfg,
        'psro_risk': psro_risk,
        'psro': psro,
        'pipeline_psro': pipeline_psro,
        'dpp_psro': dpp_psro,
        'dpp_risk': dpp_risk,
        'dpp_gamma': dpp_gamma,
        'rectified': rectified,
        'self_play': self_play,
        'gamma_list': gamma_list,
        'sampling': sampling,
        'mixed': mixed,
        'metrics': metrics,
        'method': method,
        'er': er,
        'game': game
    }

    if args.load_results:
        data_file = sampling + '_' + game + str(dim) + '.p'
        #data_file = sampling + '_' + game + str(dim) + '_old.p'
        #data_path = 'C:/Users/bruno/OneDrive/Área de Trabalho/MSc Machine Learning-DESKTOP-N01V9DD/Project/Images/20210813-180624_50_uniform_True'
        data_file = os.path.join(data_path, data_file)
        params['old_results'] = pd.read_pickle(data_file)

        # delete this after
        # results = params['old_results']
        # time_string = time.strftime("%Y%m%d-%H%M%S")
        # PATH_RESULTS = os.path.join(root_path, 'results', time_string + '_' + str(dim) + '_' + game + '_'
        #                             + sampling + '_' + str(mixed))
        # if not os.path.exists(PATH_RESULTS):
        #     os.makedirs(PATH_RESULTS)
        # plot_dist(results, params, PATH_RESULTS)

    pool = mp.Pool()
    result = pool.map(run_experiment, [(params, i) for i in range(num_experiments)])

    # this aggregates the results from all the different runs. result is a list
    my_keys = result[0].keys()
    results = {key:[] for key in my_keys}
    for r in result:
        for key, value in r.items():
            results[key].append(value)

    #create path for results
    time_string = time.strftime("%Y%m%d-%H%M%S")
    PATH_RESULTS = os.path.join(root_path, 'results', time_string + '_' + str(dim) + '_' + game + '_'
                                + sampling + '_' + str(mixed))
    if not os.path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)

    file_name = sampling + '_' + game + str(dim) + '.p'
    pickle.dump(results, open(os.path.join(PATH_RESULTS, file_name), 'wb'))
    plot_dist(results, params, PATH_RESULTS)
    plot_runtime(results, params, PATH_RESULTS)
    plot_ef(results, params, PATH_RESULTS)

    # with open(os.path.join(PATH_RESULTS, 'params.json'), 'w', encoding='utf-8') as json_file:
    #     json.dump(params, json_file, indent=4)

    # dummy experiment
    dummy_strat = run_dummy_exp(gamma_list=[ 0.01, 0.05, 0.1, 1, 10, 1000], PATH_RESULTS=PATH_RESULTS)

    def plot_error(data, label='', fill=False, obj=plt):
        if isinstance(label, list) is False:
            label = [label]

        my_array = np.array(data)
        if my_array.ndim == 3:
            my_array = np.transpose(my_array, axes=(0, 2, 1))
        else:
            my_array = np.expand_dims(my_array, axis=2)

        for i in range(my_array.shape[2]):
            array = my_array[:, :, i]
            data_mean = np.mean(array, axis=0)
            error_bars = 0.5*stats.sem(array)
            obj.plot(data_mean, label=label[i])
            if fill:
                obj.fill_between([i for i in range(data_mean.size)],
                                  np.squeeze(data_mean - error_bars),
                                  np.squeeze(data_mean + error_bars), alpha=alpha)

    alpha = .4
    titles = ['exploitability', 'cardinality', 'expected return vs RANE', 'expected return vs NE',
              'expected return vs uniform', 'variance vs RANE', 'variance vs NE', 'variance vs uniform',
              'variance exploitability', 'entropy']

    for j in range(len(titles)):  # loop over all the different charts to create
        fig_handle = plt.figure()

        if psro_risk:
            labels = ['RARisk g=' + str(gamma) for gamma in gamma_list]
            for gamma in gamma_list:
                chart_label = 'RARisk g=' + str(gamma)
                result_label = 'psro_risk_g' + str(gamma) + '_' + metrics[j]
                plot_error(results[result_label], label=chart_label, fill=True)

        if psro:
            label='PSRO'
            result_label = 'psro_' + metrics[j]
            plot_error(results[result_label], label=label, fill=True)

        if pipeline_psro:
            label='P-PSRO'
            result_label = 'pipeline_psro_' + metrics[j]
            plot_error(results[result_label], label=label, fill=True)

        if self_play:
            label = 'Self-play'
            result_label = 'self_play_' + metrics[j]
            plot_error(results[result_label], label=label, fill=True)

        if dpp_psro:
            label='DPP'
            result_label = 'dpp_psro_' + metrics[j]
            plot_error(results[result_label], label=label, fill=True)

        if dpp_risk:
            label='DPP Risk g='+str(dpp_gamma)
            result_label = 'dpp_risk_' + metrics[j]
            plot_error(results[result_label], label=label, fill=True)

        #plt.legend(loc="upper left")
        plt.legend()
        plt.title('Dim {:d}, {:}'.format(dim, titles[j]))
        plt.xlabel('iteration')

        if logscale and (j==0):
            plt.yscale('log')

        plt.savefig(os.path.join(PATH_RESULTS, 'figure_'+ str(j) + '_' + str(dim) + '_' + str(sampling)  +'.pdf'))

    # plot payoff matrix summary
    fig_handle = plt.figure()
    mat_titles = ['mat_mean', 'mat_cov', 'mat_skew', 'mat_kurt']
    mat_titles_format = ['mean', 'variance', 'skewness', 'kurtosis']
    mat_results = [np.mean(results[title]) for title in mat_titles]
    plt.bar(x=mat_titles_format, height=mat_results)
    plt.title('Dim {:d}, payoff stats'.format(dim))
    plt.savefig(os.path.join(PATH_RESULTS, 'payoffs_' + str(dim) + '_' + str(sampling) + '.pdf'))

    #plot efficient frontier
    if args.calc_ef:
        plot_ef(results, params, PATH_RESULTS)
        #plot_ef2(results, params, PATH_RESULTS)

    # plot convergence
    fig, ax = plt.subplots(nrows=2)
    #first plot
    plot_error(results['ne_ts'], label='fp_g0', fill=True, obj=ax[0])
    plot_error(results['risk_g5_ts'], label='fp_g5', fill=True, obj=ax[0])
    plot_error(results['risk_g20_ts'], label='fp_g20', fill=True, obj=ax[0])
    plot_error(results['risk_g50_ts'], label='fp_g50', fill=True, obj=ax[0])
    ax[0].set(title='strategy convergence', yscale='log', xlabel='iterations',
              ylabel='eucledian dist. b/w pol t and t-1')
    ax[0].legend(loc='upper right')
    #second
    plot_error(results['ne_diff_ts'], label='fp_g0', fill=True, obj=ax[1])
    plot_error(results['risk_g5_diff_ts'], label='fp_g5', fill=True, obj=ax[1])
    plot_error(results['risk_g20_diff_ts'], label='fp_g20', fill=True, obj=ax[1])
    plot_error(results['risk_g50_diff_ts'], label='fp_g50', fill=True, obj=ax[1])
    ax[1].set(title='br vs strategy diff', yscale='log', xlabel='iterations',
              ylabel='eucledian dist. b/w pol t and br t')
    ax[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_RESULTS, 'convergence.pdf'))

if __name__ == "__main__":
    #sampling_list = ['uniform', 't', 'normal']
    sampling_list = ['normal']
    for sampling in sampling_list:
        for dim in args.dim_list:

            run_experiments(num_experiments=args.num_experiments, num_threads=args.num_threads, iters=args.nb_iters,
                            dim=dim, lr=.5, thresh=TH, iters_nfg=args.iters_nfg,
                            psro_risk=True,
                            psro=True,
                            pipeline_psro=False,
                            rectified=False,
                            self_play=False,
                            dpp_psro=True,
                            dpp_risk=False,
                            gamma_list=args.gamma_list,
                            dpp_gamma=args.dpp_gamma,
                            sampling=sampling,
                            mixed=args.mixed,
                            metrics=metrics,
                            method=args.method,
                            er=args.er,
                            game=args.game
                            )
