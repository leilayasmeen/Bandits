import abc


class Policy:

    @abc.abstractmethod
    def choose_arm(self, spec):
        pass

    def update(self, feedback):
        self.update_model(feedback)
        self.update_metrics(feedback)

    @abc.abstractmethod
    def update_model(self, feedback):
        pass

    def update_metrics(self, feedback):
        pass


class FiniteActionPolicy(Policy):

    def __init__(self, k):
        self.k = k

    @abc.abstractmethod
    def choose_arm(self, spec) -> int:
        pass
