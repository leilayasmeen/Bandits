import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from time import time
from utils import MetricAggregator
from argparse import ArgumentParser


def create_env(file, sd):
    from envs import ContextualEnv

    # arms = np.loadtxt(file, delimiter=',')
    arms = npr.normal(size=(10, 50))  # use this to set k, then d

    return ContextualEnv(arms, sd)


def create_ols_bandit(k, d, h, q):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import DeterministicStrategy as Strategy

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    strategy = Strategy(k, q)
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_knn_bandit(k, d, h, q):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase.estimators import KNNEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import DeterministicStrategy as Strategy
    # from policies.two_phase import RandomStrategy as Strategy

    f_est = Estimator(k, d)
    a_est = Estimator(k, d)

    strategy = Strategy(k, q)
    # strategy = Strategy(k, Strategy.pow_prob(q, -1))
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_rf_bandit(k, d, h, q):  # FIXME
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase.estimators import RFEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import DeterministicStrategy as Strategy
    # from policies.two_phase import RandomStrategy as Strategy
    from optimizers import RFOpt as Opt

    f_est = Estimator(k, d)
    a_est = Estimator(k, d)

    strategy = Strategy(k, q)
    # strategy = Strategy(k, Strategy.pow_prob(q, -1))
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_random_ols_bandit(k, d, h, q):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import RandomStrategy as Strategy

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    strategy = Strategy(k, Strategy.pow_prob(q, -1))
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_lasso_bandit(k, d, h, q):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import LassoEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    #from policies.two_phase import RandomStrategy as Strategy
    from policies.two_phase import DeterministicStrategy as Strategy
    from optimizers import LassoOpt as Opt

    f_est = Estimator(k, d, Opt)
    a_est = Estimator(k, d, Opt)

    #strategy = Strategy(k, Strategy.pow_prob(q, -1))
    strategy = Strategy(k, q)
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_ucb_ols_bandit(k, d, h, b):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import GreedyStrategy as Strategy

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    strategy = Strategy(f_est, Strategy.log_budget(b))
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_greedy_bandit(k, d, h, q):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import GreedyStrategy as Strategy

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    strategy = Strategy(f_est, Strategy.log_budget(q))
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def create_thompson_sampling_imitator(k, d, h):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ThresholdSelector as Selector
    from policies.two_phase import ImitationStrategy as Strategy
    from policies.thompson import ThompsonSampling as TS

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    mus = np.zeros((k, d))
    sigmas = np.array([np.eye(d) for _ in range(k)])
    ts = TS(sd=1, mus=mus, sigmas=sigmas)

    strategy = Strategy(ts)
    selector = Selector(f_est, h)

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)
    strategy.set_imitator_policy(policy)

    return policy


def create_thompson_sampling_imitator2(k, d, b):
    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ConfBasedSelector as Selector
    from policies.two_phase import ImitationStrategy as Strategy
    from policies.thompson import ThompsonSampling as TS

    from scipy.stats import norm

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    mus = np.zeros((k, d))
    sigmas = np.array([np.eye(d) for _ in range(k)])
    ts = TS(sd=1, mus=mus, sigmas=sigmas)

    strategy = Strategy(ts)
    selector = Selector(f_est, level=lambda t: b * norm.ppf(1 - .5 / (t + 2)))

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)
    strategy.set_imitator_policy(policy)

    return policy


def create_smart_bandit(k, d, b):
    from scipy.stats import norm

    from policies.two_phase import TwoPhaseBandit
    from policies.two_phase import OlsEstimator as Estimator
    from policies.two_phase import ConfBasedSelector as Selector
    from policies.two_phase import GreedyStrategy as Strategy

    f_est = Estimator(k, d, update_always=True)
    a_est = Estimator(k, d, update_always=True)

    strategy = Strategy(f_est, Strategy.log_budget(b))
    selector = Selector(f_est, level=lambda t: b * norm.ppf(1 - .5 / (t + 2)))

    policy = TwoPhaseBandit(k, selector, strategy, f_est, a_est)

    return policy


def run_all(env, horizon):
    env.t = 0
    k, d = env.arms.shape

    h = 1.5

    algorithms = {
        'ols-q-1': create_ols_bandit(k, d, h, 1),
        # 'ols-bandit-q-2': create_ols_bandit(k, d, h, 2),
        # 'ols-q-4': create_ols_bandit(k, d, h, 4),
        # 'ols-bandit-q-8': create_ols_bandit(k, d, h, 8),
        # 'ols-bandit-q-10': create_ols_bandit(k, d, h, 10),
        # 'ols-bandit-q-20': create_ols_bandit(k, d, h, 20),
        # 'ols-bandit-q-40': create_ols_bandit(k, d, h, 40),
        # 'rand-q-4': create_random_ols_bandit(k, d, h, 4),
        # 'rand-q-40': create_random_ols_bandit(k, d, h, 40),
        # 'rand-q-60': create_random_ols_bandit(k, d, h, 60),
        # 'rand-q-80': create_random_ols_bandit(k, d, h, 80),
        # 'ucb-bandit-b-0.0001': create_ucb_ols_bandit(k, d, h, 0.0001),
        # 'ucb-b-0.0001': create_ucb_ols_bandit(k, d, h, 0.0001),
        # 'ucb-b-0.001': create_ucb_ols_bandit(k, d, h, 0.001),
        # 'ucb-b-0.01': create_ucb_ols_bandit(k, d, h, 0.01),
        # 'ucb-b-0.1': create_ucb_ols_bandit(k, d, h, 0.1),
        # 'ucb-b-1': create_ucb_ols_bandit(k, d, h, 1),
        # 'ts-imitator': create_thompson_sampling_imitator(k, d, h),
        # 'smart-ts-imitator-0.02': create_thompson_sampling_imitator2(k, d, 0.02),
        # 'smart-ts-imitator-0.1': create_thompson_sampling_imitator2(k, d, 0.1),
        # 'smart-ts-imitator-1': create_thompson_sampling_imitator2(k, d, 1),
        # 'smart-ts-imitator-10': create_thompson_sampling_imitator2(k, d, 10),
        # 'smart-ucb': create_smart_bandit(k, d, 0.1),
        # 'ts-imitator-0.1': create_thompson_sampling_imitator(k, d, 0.1),
        # 'ts-imitator-1': create_thompson_sampling_imitator(k, d, 1),
        # 'ts-imitator-10': create_thompson_sampling_imitator(k, d, 10),
        # 'ts-imitator-20': create_thompson_sampling_imitator(k, d, 20),
        # 'ts-imitator-40': create_thompson_sampling_imitator(k, d, 40),
        # 'ts-imitator-60': create_thompson_sampling_imitator(k, d, 60),
        # 'ucb-bandit-b-1': create_ucb_ols_bandit(k, d, h, 1),
        # 'greedy-bandit-q-1': create_greedy_bandit(k, d, h, 1),
        # 'greedy-bandit-q-10': create_greedy_bandit(k, d, h, 10),
        # 'greedy-bandit-q-100': create_greedy_bandit(k, d, h, 100),
        # 'thompson': create_thompson_sampling(k, d, sd),
        'lasso-bandit': create_lasso_bandit(k, d, h, 1),
        'knn': create_knn_bandit(k, d, h, 1),
        'rf': create_rf_bandit(k, d, h, 1), # FIXME
    }

    elapsed = {name: 0 for name in algorithms}

    for T in range(horizon):
        spec = env.next()

        for name, alg in algorithms.items():
            start = time()

            arm = alg.choose_arm(spec)
            fb = env.get_feedback(arm)

            alg.update(fb)

            end = time()
            elapsed[name] += end - start

    return {name: alg.metrics for name, alg in algorithms.items()}


def plot_confidence_bound(band, label):
    y, _, se = band
    x = np.arange(0, len(y))
    plt.fill_between(x, y - se, y + se, alpha=0.2)
    plt.plot(x, y, label=label)


def __main__(file, sd, horizon, run, seed):
    env = create_env(file, sd)

    aggs = {}

    for i in range(run):
        print(f'round {i} started...')
        npr.seed(seed + i)

        metrics = run_all(env, horizon)
        for name, metric in metrics.items():
            if name not in aggs:
                aggs[name] = tuple(MetricAggregator() for _ in range(4))

            f_regs = np.cumsum(metric.f_regs)
            a_regs = np.cumsum(metric.a_regs)
            t_regs = f_regs + a_regs

            aggs[name][0].aggregate(f_regs)
            aggs[name][1].aggregate(a_regs)
            aggs[name][2].aggregate(t_regs)

            # costs = metric.a_regs
            # filter = lambda i: not metric.forced[i]
            # aggs[name][3].aggregate(costs, filter)

            f_count = np.cumsum(metric.forced)
            aggs[name][3].aggregate(f_count)

            print(f'Average selected-set for {name}: {np.mean([len(x) for x in metric.selections])}')

    for name, agg in aggs.items():
        for i, cls in enumerate(('forced', 'all', 'total', 'avg-cost')):
            plt.subplot(2, 2, i + 1)
            plot_confidence_bound(agg[i].confidence_band(), f'{name}')

    for i, title in enumerate(('forced-sampling regret',
                               'all-sampling regret',
                               'cumulative regret',
                               'the number of times that exploration happens')):
        plt.subplot(2, 2, i + 1)
        plt.title(title)
        plt.legend()

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arms', type=str)  # , required=True)
    parser.add_argument('--sd', type=float, default=1.)
    parser.add_argument('--run', type=int, default=5)  # change this to 1000 before obtaining final results
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=314159265)

    args = parser.parse_args()

    __main__(args.arms, args.sd, args.horizon, args.run, args.seed)
