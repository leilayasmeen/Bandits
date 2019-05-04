import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import pairwise as distance_metrics

class KNNEst:

    def __init__(self):
        pass

    def optimize(self, xs, ys, kn):

        # Scale to N(0,1)
        scaler = StandardScaler()
        scaler.fit(xs)
        xsscaled = scaler.transform(X=xs)

        #rbfdist = distance_metrics.rbf_kernel(xsscaled)

        # calculate "kn"-nearest neighbor groups
        # neigh = KNR(n_neighbors=kn, weights='distance', metric=rbfdist)
        neigh = KNR(n_neighbors=kn, weights='distance')
        fit = neigh.fit(xsscaled,ys)
        y1 = neigh.predict(xsscaled)  # FIXME: you predict on the same data that you trained the KNN model on?

        # e1 = time.time()

        r1 = np.linalg.norm(ys - y1) ** 2
        # l1 = r1 + lamda * np.linalg.norm(b1, ord=1) # FIXME: what is loss? Sum of distances to nearest neighbors?
        print('KNN:')
        print('rss: %.5f' % r1)
        # print('loss: %.5f' % l1)
        # print('l1 dist: %.5f' % np.linalg.norm(b1 - b))
        # print('time: %f' % (e1 - s1))
        print()

        return fit, r1, y1

KNNOpt = KNNEst

def __main__():  # Set the estimated y-value for each x observation as the average of its k-nearest neighbors
    # import time

    # Reference on KNN regression...
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    np.random.seed(3141592657)
    d, s = 10, 2
    n = 1000

    b = np.random.rand(d)
    m = np.random.choice(range(d), d - s, replace=False)
    b[m] = 0

    xs = np.random.rand(n, d)
    ys = np.dot(xs, b) + np.random.randn(n)

    # Scale to N(0,1)
    scaler = StandardScaler()
    scaler.fit(xs)
    xsscaled = scaler.transform(xs)

    opt1 = KNNEst()
    # s1 = time.time()
    neigh = opt1.optimize(xsscaled, ys, kn=5)
    y1 = neigh[1]  # FIXME: you predict on the same data that you trained the KNN model on?

    # e1 = time.time()

    r1 = np.linalg.norm(ys - y1) ** 2
    # l1 = r1 + lamda * np.linalg.norm(b1, ord=1) # FIXME: what would loss be for KNN?
    print('KNN:')
    print('rss: %.5f' % r1)
    # print('loss: %.5f' % l1)
    # print('l1 dist: %.5f' % np.linalg.norm(b1 - b))
    # print('time: %f' % (e1 - s1))
    print()


if __name__ == '__main__':
    __main__()
