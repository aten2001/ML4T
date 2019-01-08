import numpy as np


class DTLearner:

    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size

        self.tree = None

    def build_tree(self, X, y):
        if X.shape[0] <= self.leaf_size:
            return np.array([[None, np.mean(y), None, None]])

        if np.all(y == y[0]):
            return np.array([[None, y[0], None, None]])

        feature = np.argmax([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])

        split_val = np.median(X[:, feature])

        mask = X[:, feature] <= split_val

        if np.all(mask):
            return np.array([[None, np.mean(y), None, None]])

        left = self.build_tree(X[mask], y[mask])
        right = self.build_tree(X[~mask], y[~mask])

        root = [feature, split_val, 1, left.shape[0] + 1]

        return np.vstack((root, left, right))

    def addEvidence(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        tree = self.tree

        i = 0

        while True:
            feature, split_val, left, right = tree[int(i), :]

            if feature is None:
                return split_val

            if X[int(feature)] <= split_val:
                i += left
            else:
                i += right

    def query(self, X):
        res = list()

        for i in X:
            res.append(self.predict(i))

        return res
