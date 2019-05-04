import abc

from typing import Callable
from operator import itemgetter

from envs.contextual import ContextualSpec as CtxSpec
from envs.contextual import ContextualFeedback as CtxFb

from .estimators import BanditEstimator as Estimator
from .estimators import ConfidenceEstimator as ConfEstimator


class Selector:

    @abc.abstractmethod
    def select(self, spec: CtxSpec):
        pass

    def update(self, feedback: CtxFb):
        pass


class ThresholdSelector(Selector):

    def __init__(self, est: Estimator, h: float):
        self.h = h
        self.est = est

    def select(self, spec: CtxSpec):
        mean_rews = self.est.predict_rewards(spec)
        max_rew = max(mean_rews)

        return [i for i, rew in enumerate(mean_rews) if rew > max_rew - self.h]


class ConfBasedSelector(Selector):

    def __init__(self, est: ConfEstimator, level: Callable[[int], float]):
        self.est = est
        self.level = level

    def select(self, spec: CtxSpec):
        level = self.level(spec.t)
        cis = self.est.conf_ints(spec, level)
        max_lower_bound, _ = max(cis, key=itemgetter(0))

        return [i for i, ci in enumerate(cis) if ci[1] >= max_lower_bound]
