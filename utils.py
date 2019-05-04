import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import scipy.linalg as scl

from typing import List


def is_power_two(x):
    while x > 1:
        if x % 2 == 0:
            x /= 2
        else:
            return False
    return True


def sqrt_sym(x):
    y = scl.sqrtm(x)
    return np.real(y)


def argmax(arr: List) -> int:
    max_val = max(arr)

    for i, val in enumerate(arr):
        if val == max_val:
            return i


def decompress_obs(xx, xy, yy):
    d = xx.shape[0]

    xs = np.zeros((d + 1, d))
    ys = np.zeros((d + 1,))

    xs[:d, :] = scl.sqrtm(xx)
    ys[:d] = npl.solve(xs[:d, :], xy)
    ys[d] = np.maximum(yy - np.sum(ys ** 2), 0) ** 0.5

    return xs, ys


class DataStore:

    def __init__(self, d, every=3):
        self.d = d

        self.tol = every * d

        self.xx = np.zeros((d, d))
        self.xy = np.zeros((d,))
        self.yy = 0

        self.xs = []
        self.ys = []

        self.empty = True
        self.dirty = False

    def add_obs(self, x, y):
        self.xx += np.outer(x, x)
        self.xy += x * y
        self.yy += y ** 2

        self.empty = False

        if len(self.xs) <= self.tol:
            self.xs.append(x)
            self.ys.append(y)
        else:
            self.dirty = True

    def get_obs(self):
        if self.dirty:
            xs, ys = decompress_obs(self.xx, self.xy, self.yy)

            self.xs = xs.tolist()
            self.ys = ys.tolist()

            self.dirty = False

        return np.array(self.xs), np.array(self.ys)

    def get_xs(self):

        return self.get_obs()[0]

    def get_ys(self):

        return self.get_obs()[1]


class MetricAggregator:

    def __init__(self):
        self.m0 = []
        self.m1 = []
        self.m2 = []

    def confidence_band(self):
        m0 = np.array(self.m0)
        m1 = np.array(self.m1)
        m2 = np.array(self.m2)

        m0 = np.maximum(m0, 1)

        mean = m1 / m0
        var = (m2 - m1 ** 2 / m0) / (m0 - 1)
        sd = var ** 0.5
        se = (var / m0) ** 0.5

        return mean, sd, se

    def aggregate(self, xs, filter=lambda _: True):
        self._ensure_len(len(xs))

        for i, x in enumerate(xs):
            if filter(i):
                self.m0[i] += 1
                self.m1[i] += x
                self.m2[i] += x ** 2

    def _ensure_len(self, n):
        dn = n - len(self.m0)

        if dn > 0:
            self.m0 += [0] * dn
            self.m1 += [0] * dn
            self.m2 += [0] * dn


def __main__():
    npr.seed(314159265)

    d = 50
    n = 102

    xs = npr.random((n, d))
    ys = npr.random((n, ))

    ds = DataStore(d)
    for i in range(n):
        ds.add_obs(xs[i], ys[i])

        dxs, dys = ds.get_obs()

        b = npr.random((d, ))
        expect = npl.norm(ys[:(i + 1)] - np.dot(xs[:(i + 1)], b))
        actual = npl.norm(dys - np.dot(dxs, b))

        assert np.abs((actual - expect) / expect) < 0.00001

    print('DataStore passed all tests.')


if __name__ == '__main__':
    __main__()
