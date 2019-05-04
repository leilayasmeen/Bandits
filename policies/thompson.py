import numpy as np
import numpy.random as npr

from utils import sqrt_sym
from .base import FiniteActionPolicy


class ThompsonSampling(FiniteActionPolicy):

    def __init__(self, sd, mus, sigmas):
        super().__init__(mus.shape[0])

        self.t = 0
        self.sd = sd
        self.mus = mus
        self.sigmas = sigmas
        self.sqrts = np.zeros(sigmas.shape)
        for k in range(sigmas.shape[0]):
            self.sqrts[k] = sqrt_sym(sigmas[k])

    def update_model(self, feedback):
        arm = feedback.arm
        ctx = feedback.ctx
        rew = feedback.rew

        x = ctx / self.sd
        y = rew / self.sd

        sigma_old = self.sigmas[arm]

        sigma_x = sigma_old @ x
        sigma_upd = np.outer(sigma_x, sigma_x) / (1 + (x @ sigma_x))
        sigma_new = sigma_old - sigma_upd

        mu_old = self.mus[arm]
        mu_new = (sigma_new @ x) * y + mu_old - sigma_x / (1 + (x @ sigma_x)) * (x @ mu_old)

        self.sigmas[arm] = sigma_new
        self.sqrts[arm] = sqrt_sym(sigma_new)
        self.mus[arm] = mu_new

    def choose_arm(self, spec):
        ctx = spec.ctx

        d = ctx.shape[0]
        muhs = [self.mus[k] + np.dot(self.sqrts[k], npr.normal(size=d))
                for k in range(self.mus.shape[0])]
        rews = [np.dot(muh, ctx) for muh in muhs]

        return np.argmax(rews)
