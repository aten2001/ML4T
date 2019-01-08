import numpy as np


class BagLearner:

    def __init__(self, learner, kwargs, bags, boost=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost

        self.models = None

    def addEvidence(self, X, y):
        learner = self.learner
        bags = self.bags
        kwargs = self.kwargs

        models = []

        for i in range(bags):
            mask = np.random.choice(np.arange(X.shape[0]), X.shape[0])
            model = learner(**kwargs)
            model.addEvidence(X[mask], y[mask])
            models.append(model)

        self.models = models

    def query(self, X):
        models = self.models
        bags = self.bags

        res = np.empty((bags, X.shape[0]))

        for i in range(bags):
            res[i, :] = models[i].query(X)

        return np.mean(res, axis=0)
