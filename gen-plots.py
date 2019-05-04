import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def __main__(input, output):

    files = [os.path.join(input, f) for f in os.listdir(input)]
    dfs = [pd.read_csv(f) for f in files]

    m0 = {}
    m1 = {}
    m2 = {}
    for df in dfs:
        for name, group in df.groupby('name'):
            if name not in m0:
                l = len(group)
                m0[name] = np.zeros((l,))
                m1[name] = np.zeros((l,))
                m2[name] = np.zeros((l,))

            cum_regret = np.cumsum(group.regret)
            m0[name] += 1
            m1[name] += cum_regret
            m2[name] += cum_regret ** 2

    for name in m0:
        n = m0[name]
        s1 = m1[name]
        s2 = m2[name]

        t = np.arange(len(n)) + 1

        mean = s1 / n
        var = (s2 - s1 ** 2 / n) / (n * (n - 1))

        lower = mean - var ** 0.5
        upper = mean + var ** 0.5

        plt.plot(t, mean, label=name)
        plt.fill_between(t, lower, upper, alpha=0.2)

    plt.legend()

    if output is None:
        plt.show()
    else:
        plt.savefig(output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    __main__(args.input, args.output)
