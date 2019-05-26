import abc

import numpy as np
import numpy.linalg as npl

from utils import is_power_two
from utils import DataStore

from envs.contextual import ContextualSpec as CtxSpec
from envs.contextual import ContextualFeedback as CtxFb

from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor as RFReg

from sklearn.model_selection import GridSearchCV


class BanditEstimator:

    @abc.abstractmethod
    def __init__(self, k, d):
        self.k = k
        self.d = d

    @abc.abstractmethod
    def add_obs(self, feedback: CtxFb):
        pass

    @abc.abstractmethod
    def predict_reward(self, arm: int, spec: CtxSpec):
        pass

    def predict_rewards(self, spec: CtxSpec):

        return [self.predict_reward(arm, spec) for arm in range(self.k)]


class LinearEstimator(BanditEstimator):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_reward(self, arm: int, spec: CtxSpec):
        ctx = spec.ctx

        return ctx @ self[arm]

    @abc.abstractmethod
    def __getitem__(self, arm):
        pass

    def __len__(self):
        return self.k

    def __iter__(self):
        def gen():
            for i in range(self.k):
                yield self[i]

        return gen()


class ConfidenceEstimator(BanditEstimator):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def conf_int(self, arm: int, spec: CtxSpec, level: float):
        pass

    def conf_ints(self, spec: CtxSpec, level: float):

        return [self.conf_int(arm, spec, level) for arm in range(self.k)]


class OlsEstimator(LinearEstimator, ConfidenceEstimator):

    def __init__(self, k, d, update_always=False):
        super().__init__(k, d)

        self.obs = [DataStore(d) for _ in range(k)]
        self.arms = np.zeros((k, d))
        self.dirty = np.zeros((k,))
        self.update_always = update_always

    def add_obs(self, feedback: CtxFb):
        ctx = feedback.ctx
        arm = feedback.arm
        rew = feedback.rew

        self.obs[arm].add_obs(ctx, rew)
        self.dirty[arm] = 1

    def __getitem__(self, arm):
        if self.dirty[arm] == 1:
            xs, ys = self.obs[arm].get_obs()
            if self.update_always or is_power_two(len(xs)):
                self.arms[arm] = npl.lstsq(xs, ys, rcond=-1)[0]
                self.dirty[arm] = 0
        return self.arms[arm]

    def conf_int(self, arm: int, spec: CtxSpec, level: float):
        ctx = spec.ctx
        xs = self.obs[arm].get_xs()

        if xs.shape == (0, ):
            return - np.inf, np.inf

        if xs.shape[0] < xs.shape[1]:
            return - np.inf, np.inf

        xx = xs.T @ xs

        mid = self.predict_reward(arm, spec)
        scale = ctx @ npl.solve(xx, ctx)

        left = mid - scale * level
        right = mid + scale * level

        return left, right


class LassoEstimator(LinearEstimator):

    def __init__(self, k, d, opt_class):
        super().__init__(k, d)

        self.obs = [DataStore(d) for _ in range(k)]
        self.opts = [opt_class() for _ in range(k)]
        self.arms = np.zeros((k, d))
        self.dirty = np.zeros((k, ))

    def add_obs(self, feedback: CtxFb):
        arm = feedback.arm
        ctx = feedback.ctx
        rew = feedback.rew

        self.obs[arm].add_obs(ctx, rew)
        self.dirty[arm] = 1

    def __getitem__(self, arm):
        if self.dirty[arm]:
            xs, ys = self.obs[arm].get_obs()
            self.arms[arm], _ = self.opts[arm].optimize(xs, ys, lamda = 25) # this calculates estimate for that arm
            self.dirty[arm] = 0
        return self.arms[arm]


class KNNEstimator(BanditEstimator):

    def __init__(self, k, d):
        super().__init__(k, d)

        self.xs = [[] for _ in range(k)]
        self.ys = [[] for _ in range(k)]

    def add_obs(self, feedback: CtxFb):
        arm = feedback.arm
        ctx = feedback.ctx
        rew = feedback.rew

        self.xs[arm].append(ctx)
        self.ys[arm].append(rew)

    def predict_reward(self, arm: int, spec: CtxSpec):
        xs = self.xs[arm]
        ys = self.ys[arm]

        if len(xs) <= 5:
            return 0
        else:
            ctx = spec.ctx

            neigh = KNR(n_neighbors=5, weights='distance')

            neigh.fit(xs, ys)

            y1 = neigh.predict(np.array([ctx]))[0]

            return y1

class RFEstimator(BanditEstimator):

    def __init__(self, k, d):
        super().__init__(k, d)

        self.xs = [[] for _ in range(k)]
        self.ys = [[] for _ in range(k)]

    def add_obs(self, feedback: CtxFb):
        arm = feedback.arm
        ctx = feedback.ctx
        rew = feedback.rew

        self.xs[arm].append(ctx)
        self.ys[arm].append(rew)

    def predict_reward(self, arm: int, spec: CtxSpec):
        xs = self.xs[arm]
        ys = self.ys[arm]

        if len(xs) <= 5:
            return 0
        else:
            ctx = spec.ctx

            regr = RFReg()

            regr.fit(xs, ys)

            y1 = regr.predict(np.array([ctx]))[0]

            return y1



class KNNCVEstimator(KNNEstimator):

    def __init__(self, k, d, j=5, knlen=10):
        super().__init__(k, d)

        self.xs = [[] for _ in range(k)]
        self.ys = [[] for _ in range(k)]

    def add_obs(self, feedback: CtxFb):
        arm = feedback.arm
        ctx = feedback.ctx
        rew = feedback.rew

        self.xs[arm].append(ctx)
        self.ys[arm].append(rew)

    def cv_estimator(self):

        knntest = KNR()

        # dict of values of kn to test
        cvgrid = {‘n_neighbors’: np.linspace(0, len(xs)/10), num = min(5,20), dtype=np.uint8} # change to size of dataset / 10 in regularly spaced increments

        # use gridsearch to test all values for n_neighbors
        knn_gridsearch= GridSearchCV(knntest, cvgrid, cv=j)

        # fit model to data
        knn_gridsearch.fit(xs, ys)

        # best n_neighbors value
        knstar = knn_gridsearch.best_params_

        # https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

        return knstar

    def predict_reward(self, arm: int, spec: CtxSpec):
        xs = self.xs[arm]
        ys = self.ys[arm]

        if len(xs) <= 10:
            return 0
        else:
            ctx = spec.ctx

            knstar = cv_estimator() # FIXME: should I call the function here?

            neigh = KNR(n_neighbors=knstar, weights='distance')

            neigh.fit(xs, ys)

            y1 = neigh.predict(np.array([ctx]))[0]

            return y1
