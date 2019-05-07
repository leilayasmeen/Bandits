from envs.contextual import ContextualSpec as CtxSpec
from envs.contextual import ContextualFeedback as CtxFb

from policies.base import FiniteActionPolicy

from policies.two_phase.selectors import Selector
from policies.two_phase.estimators import BanditEstimator as Estimator
from policies.two_phase.strategies import ForcedSamplingStrategy as Strategy


class TwoPhaseBandit(FiniteActionPolicy): # use to define bandit

    def __init__(self, k,
                 selector: Selector,
                 strategy: Strategy,
                 f_est: Estimator,
                 a_est: Estimator,
                 strict: bool = False): # False = add all samples, not just forced.

        super().__init__(k)

        self.strict = strict

        self.selector = selector
        self.strategy = strategy
        self.selection = []

        self.f_est = f_est
        self.a_est = a_est

        self.metrics = TwoPhaseBanditMetric()
        self.forced_arm = None

    def update_model(self, feedback: CtxFb) -> bool:
        forced = (self.forced_arm == feedback.arm)

        if forced:
            self.f_est.add_obs(feedback)

        if not forced or not self.strict:
            self.a_est.add_obs(feedback)

        self.strategy.update(feedback)
        self.selector.update(feedback)

        return forced

    def update_metrics(self, feedback: CtxFb) -> None:
        forced = (self.forced_arm == feedback.arm)

        self.metrics.add_obs(feedback, forced)
        self.metrics.add_selection(self.selection)

    def choose_arm(self, spec: CtxSpec) -> int:

        self.forced_arm = self.strategy.pick_arm(spec)

        # force-sampling phase
        if self.forced_arm is not None:
            return self.forced_arm

        return self.choose_arm_by_exploitation(spec)

    def choose_arm_by_exploitation(self, spec: CtxSpec) -> int:
        self.selection = self.selector.select(spec)

        return max(self.selection,
                   key=lambda i: self.a_est.predict_reward(i, spec))


class TwoPhaseBanditMetric:

    def __init__(self):
        self.arms = []
        self.forced = []
        self.f_regs = []
        self.a_regs = []
        self.selections = []

    def add_obs(self, feedback: CtxFb, forced: bool):
        arm = feedback.ctx
        reg = feedback.regret

        self.arms.append(arm)
        self.forced.append(forced)

        f_reg, a_reg = (reg, 0) if forced else (0, reg)

        self.f_regs.append(f_reg)
        self.a_regs.append(a_reg)

    def add_selection(self, sel):
        self.selections.append(sel)
