import numpy as np
import numpy.random as npr

from .base import Environment, Feedback, Spec


class ContextualEnv(Environment):

    def __init__(self, arms, sd):
        super().__init__()

        self.arms = arms
        self.sd = sd

        self.max_rew = None
        self.mean_rews = None

    @property
    def k(self):
        return self.arms.shape[0]

    @property
    def d(self):
        return self.arms.shape[1]

    @property
    def regrets(self):
        return self.max_rew - self.mean_rews

    def create_spec(self):
        ctx = npr.randn(self.d)  # Note to self: K x d data generated here. K = number of arms, d = dimensions

        self.mean_rews = self.arms.dot(ctx) # FIXME: adjust how reward generated - e.g. relu. np.heaviside(ctx, 0)
        self.max_rew = np.max(self.mean_rews)

        return ContextualSpec(self.t, ctx)

    def get_feedback(self, arm, spec=None):
        if spec is None:
            spec = self.spec
            mean_rews = self.mean_rews
            max_rew = self.max_rew

        else:
            mean_rews = self.arms.dot(spec.ctx)
            max_rew = np.max(mean_rews)

        mean = mean_rews[arm]
        noise = npr.randn() * self.sd
        rew = mean + noise # FIXME: is this supposed to be adjusted if I want non-linear rewards?

        return ContextualFeedback(spec, arm, rew, noise, max_rew)


class ContextualSpec(Spec):

    def __init__(self, t, ctx):
        super().__init__(t)

        self.ctx = ctx


class ContextualFeedback(Feedback):

    def __init__(self, spec, arm, rew, noise, max_rew):
        super().__init__(spec, arm, rew)

        self.noise = noise
        self.max_rew = max_rew

    @property
    def t(self):
        return self.spec.t

    @property
    def ctx(self):
        return self.spec.ctx

    @property
    def mean_rew(self):
        return self.rew - self.noise

    @property
    def regret(self):
        return self.max_rew - self.mean_rew

    def __repr__(self):
        return f'CtxFb(arm={self.arm}, reg={self.regret}, noise={self.noise}, mean={self.mean_rew})'
