import numpy as np


class GBayes:

    def __init__(self):
        pass

    def fit(self, features, classes):

        numtrainData = len(features)
        numClasses = len(np.unique(classes))

        nums = np.zeros(numClasses)

        u = np.zeros((numClasses, features.shape[1]), float)
        var = np.zeros((numClasses, features.shape[1]), float)

        for i in range(numtrainData):
            nums[classes[i]] += 1
            u[classes[i]] += features[i]

        u = u / nums[:, np.newaxis]

        for i in range(numtrainData):
            var[classes[i]] += (features[i] - u[classes[i]])**2

        var = var / nums[:, np.newaxis]

        self.u = u
        self.var = var
        self.numClasses = len(np.unique(classes))

    def predict(self, f):

        p = np.zeros((self.numClasses), float)
        r = []

        for x in f:
            for i in range(len(p)):
                p[i] = (1.0 / np.sqrt(2 * np.pi * (np.linalg.norm(self.var[i])))) * np.exp(-(np.linalg.norm((x - self.u[i])**2)) / (2 * np.linalg.norm(self.var[i])))

            r.append(np.argmax(p))

        return np.array(r)

    def predict_proba(self, f):
        p = np.zeros((self.numClasses), float)
        r = []
        
        for x in f:
            for i in range(len(p)):
                p[i] = (1.0 / np.sqrt(2 * np.pi * (np.linalg.norm(self.var[i])))) * np.exp(-(np.linalg.norm((x - self.u[i])**2)) / (2 * np.linalg.norm(self.var[i])))
            r.append(p)
        return r
