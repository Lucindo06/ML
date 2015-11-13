import numpy as np


class Kmeans:

    def __init__(self, k, max_iter=1000):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X, y=None):
        n_instances = X.shape[0]
        n_features = X.shape[1]
        k = self.k
        max_iter = self.max_iter

        # on récupère le min et le max de nos données
        min_f = np.min(X, axis=0)
        max_f = np.max(X, axis=0)

        # Initialisation des cenres
        centres = np.random.random(
            size=(k, n_features)) * (max_f - min_f) + min_f
        dernierCentres = np.zeros(centres.shape)

        iteration = 0
        while np.linalg.norm(dernierCentres - centres) < 1e-5 \
                and iteration < max_iter:
            iteration += 1
            dernierCentres = centres.copy()
            # Compute the distances
            dist = np.empty((n_instances, k))
            for i in range(k):
                dist[:, i] = np.linalg.norm(X[:] - centres[i], axis=1)

            # Get the new cluster centers
            cluster = np.argmin(dist, axis=1)
            # cluster = np.transpose(cluster*np.ones((1,self.dataNums)))
            for i in range(k):
                currentCluster = np.where(cluster == i)[0]
                if currentCluster.shape[0] > 0:
                    centres[i] = np.sum(
                        X[currentCluster], axis=0) / currentCluster.shape[0]

        self.centres = centres

    def predict(self, X):
        n_instances = X.shape[0]
        k = self.k
        centres = self.centres
        # On calcul la distance avec les centres
        dist = np.empty((n_instances, k))
        for i in range(k):
            dist[:, i] = np.linalg.norm(X[:] - centres[i], axis=1)

        # On retourne le cluster
        cluster = dist.argmin(axis=1)

        return cluster
