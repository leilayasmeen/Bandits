import abc
import math

from operator import itemgetter
from typing import Union

import numpy as np
import numpy.random as npr

from envs.contextual import ContextualSpec as CtxSpec
from envs.contextual import ContextualFeedback as CtxFb

from policies.base import FiniteActionPolicy as Policy
from utils import is_power_two


class ForcedSamplingStrategy:

    @abc.abstractmethod
    def pick_arm(self, spec: CtxSpec) -> Union[int, None]:
        pass

    def update(self, feedback: CtxFb):
        pass


class DeterministicStrategy(ForcedSamplingStrategy):

    def __init__(self, k, q):
        self.k = k
        self.q = q

    def pick_arm(self, spec: CtxSpec) -> Union[int, None]:
        t = spec.t

        if is_power_two(t // (self.k * self.q) + 1):
            return (t % (self.k * self.q)) // self.q

        return None


class RandomStrategy(ForcedSamplingStrategy):

    def __init__(self, k, prob):
        self.k = k
        self.prob = prob

    def pick_arm(self, spec: CtxSpec) -> Union[int, None]:
        t = spec.t

        if npr.uniform() < self.prob(t):
            return npr.randint(0, self.k)

        return None

    @staticmethod
    def fixed_prob(p):
        return RandomStrategy.pow_prob(p, 0)

    @staticmethod
    def pow_prob(r, a):
        def _prob(t):
            return np.clip(r * (t + 1) ** a, 0, 1)

        return _prob


class GreedyStrategy(ForcedSamplingStrategy):

    def __init__(self, estimator, budget):
        self.estimator = estimator
        self.budget = budget

    def pick_arm(self, spec: CtxSpec) -> Union[int, None]:
        t = spec.t
        ctx = spec.ctx

        values = [ctx.dot(ds.xx).dot(ctx) / ctx.dot(ctx) for ds in self.estimator.obs]

        arm, val = min(enumerate(values), key=itemgetter(1))

        if val < self.budget(t):
            return arm

        return None

    @staticmethod
    def fixed_budget(b):
        def _budget(t):
            return b

        return _budget

    @staticmethod
    def log_budget(b):
        def _budget(t):
            return b * math.log(t + 1)

        return _budget

    @staticmethod
    def sqrt_budget(b):
        def _budget(t):
            return b * math.sqrt(t + 1)

        return _budget


class ImitationStrategy(ForcedSamplingStrategy):

    def __init__(self, policy: Policy):
        self.policy = policy
        self.imitator = None

    def set_imitator_policy(self, imitator: Policy):
        self.imitator = imitator

    def pick_arm(self, spec):
        arm = self.policy.choose_arm(spec)

        if arm == self.imitator.choose_arm_by_exploitation(spec):
            return None

        return arm

    def update(self, feedback: CtxFb):
        self.policy.update(feedback)
