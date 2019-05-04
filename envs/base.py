import abc


class Environment:

    def __init__(self):
        self.t = 0
        self.spec = None

    def next(self):
        self.spec = self.create_spec()

        self.t += 1

        return self.spec

    @abc.abstractmethod
    def create_spec(self):
        pass

    @abc.abstractmethod
    def get_feedback(self, arm, spec=None):
        pass


class Feedback:

    def __init__(self, spec, arm, rew):
        self.spec = spec
        self.arm = arm
        self.rew = rew


class Spec:

    def __init__(self, t):
        self.t = t
