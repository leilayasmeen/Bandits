import abc

import numpy as np
import numpy.linalg as npl

from utils import is_power_two
from utils import DataStore

from envs.contextual import ContextualSpec as CtxSpec
from envs.contextual import ContextualFeedback as CtxFb


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


class LinearEstimator(BanditEstimator):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_reward(self, arm: int, spec: CtxSpec):
        ctx = spec.ctx

        return ctx @ self[arm]


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
            self.arms[arm], _ = self.opts[arm].optimize(xs, ys, lamda=25) # this calculates estimate for that arm
            self.dirty[arm] = 0
        return self.arms[arm]


class KNNEstimator(LinearEstimator):  # FIXME

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
            self.arms[arm], _ = self.opts[arm].optimize(xs, ys, kn = 5) # FIXME: Unless kn = 1
            self.dirty[arm] = 0
        return self.arms[arm]