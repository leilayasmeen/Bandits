import numpy as np
# import cvxpy as cvx
import numpy.linalg as npl
import sklearn.linear_model as skl


# class CvxLasso:
#
#     def __init__(self):
#         pass
#
#     def optimize(self, xs, ys, lamda):
#         p = xs.shape[1]
#         b = cvx.Variable((p, ))
#
#         rss = cvx.sum_squares(ys - xs * b)
#         reg = lamda * cvx.norm(b, p=1)
#         loss = rss + reg
#
#         obj = cvx.Minimize(loss)
#         prob = cvx.Problem(obj)
#         prob.solve()
#
#         return b.value, [obj.value, loss.value]


class SkLasso:
    def __init__(self):
        self.lasso = skl.Lasso(warm_start=True,
                               fit_intercept=False,
                               copy_X=False,
                               tol=1e-5,
                               max_iter=100000)

    def optimize(self, xs, ys, lamda):
        n_obs = xs.shape[0]

        self.lasso.set_params(alpha=lamda / (2 * n_obs))

        self.lasso.fit(xs, ys)

        coef = self.lasso.coef_
        res = ys - np.dot(xs, coef)

        rss = np.inner(res, res)
        reg = npl.norm(coef, ord=1)

        return coef, [rss + lamda * reg, rss]


LassoOpt = SkLasso


def __main__():
    import time
    import numpy.random as npr
    import numpy.linalg as npl

    npr.seed(3141592657)
    d, s = 200, 30
    n = 1000

    b = npr.rand(d)
    m = npr.choice(range(d), d - s, replace=False)
    b[m] = 0

    xs = npr.rand(n, d)
    ys = np.dot(xs, b) + npr.randn(n)

    lamda = 10

    opt1 = CvxLasso()
    s1 = time.time()
    b1, _ = opt1.optimize(xs, ys, lamda)
    e1 = time.time()

    r1 = npl.norm(ys - np.dot(xs, b1)) ** 2
    l1 = r1 + lamda * npl.norm(b1, ord=1)
    print('CvxLasso:')
    print('rss: %.5f' % r1)
    print('loss: %.5f' % l1)
    print('l1 dist: %.5f' % npl.norm(b1 - b))
    print('time: %f' % (e1 - s1))
    print()

    opt2 = SkLasso()
    s2 = time.time()
    b2, _ = opt2.optimize(xs, ys, lamda)
    e2 = time.time()
    r2 = npl.norm(ys - np.dot(xs, b2)) ** 2
    l2 = r2 + lamda * npl.norm(b2, ord=1)
    print('SkLasso:')
    print('rss: %.5f' % r2)
    print('loss: %.5f' % l2)
    print('l1 dist: %.5f' % npl.norm(b2 - b))
    print('time: %f' % (e2 - s2))
    print()

    print('cvx-loss / sk-loss: %.5f' % (l1 / l2))


if __name__ == '__main__':
    __main__()
