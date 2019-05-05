import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RFEst:

    def __init__(self):
        pass

    def optimize(self, xs, ys, max_depth=5, random_state=0, n_estimators=100):

        regr = RandomForestRegressor()

        estfit = regr.fit(xs,ys, max_depth, random_state, n_estimators)
        y1 = regr.pred(xs, ys)
        r1 = np.linalg.norm(ys - y1) ** 2

        return estfit, r1, y1

RFOpt = RFEst

def __main__():
    import time

    np.random.seed(3141592657)
    d, s = 5, 2
    n = 1000

    b = np.random.rand(d)
    m = np.random.choice(range(d), d - s, replace=False)
    b[m] = 0

    xs = np.random.rand(n, d)
    ys = np.dot(xs, b) + np.random.randn(n)

    opt1 = RFEst()
    s1 = time.time()
    estimated_fit, r1, y1 = opt1.optimize(xs, ys)

    e1 = time.time()

    r1 = np.linalg.norm(ys - y1) ** 2
    print('Random Forest rss: %.5f' % r1)
    print('time: %f' % (e1 - s1))
    print()

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

if __name__ == '__main__':
    __main__()
