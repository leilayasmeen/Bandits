import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFReg


class RFEst:

    def __init__(self):
        pass

    def optimize(self, xs, ys, max_depth=5, random_state=0, n_estimators=100):

        regr = RandomForestRegressor(max_depth, random_state, n_estimators)

        estfit = regr.fit(xs,ys)
        estpred = regr.pred(xs,ys)

        return estfit, estpred

RFOpt = RFEst

def __main__(): # Set the estimated y-value for each x observation using the random forest
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
    estimated_fit, y1 = opt1.optimize(xs, ys, max_depth, random_state, n_estimators)
    # TOCHECK: you predict on the same data that you trained the RF model on?

    e1 = time.time()

    r1 = np.linalg.norm(ys - y1) ** 2
    #l1 = r1 + lamda * np.linalg.norm(b1, ord=1) # TOCHECK: what would loss be for KNN?
    print('KNN:')
    print('rss: %.5f' % r1)
    #print('loss: %.5f' % l1)
    #print('l1 dist: %.5f' % np.linalg.norm(b1 - b))
    print('time: %f' % (e1 - s1))
    print()

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

if __name__ == '__main__':
    __main__()
