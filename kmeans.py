# coding: utf-8
import numpy as np


class Kmeans:

    def __init__(self, k, maxIter=100):

        # self.data = data
        # self.dataNums = np.shape(data)[0]
        # self.dataDim = np.shape(data)[1]
        self.k = k
        self.maxIter = maxIter

    def fit(self, data, label=None):

        # on récupère le min et le max de nos données
        self.data = data
        self.dataNums = np.shape(data)[0]
        self.dataDim = np.shape(data)[1]
        min_f = np.min(self.data, axis=0)
        max_f = np.max(self.data, axis=0)

        # Initialisation des cenres
        self.centres = np.random.rand(
            self.k, self.dataDim) * (max_f - min_f) + min_f
        dernierCentres = np.zeros(self.centres.shape)

        iteration = 0

        while np.sum(dernierCentres - self.centres) != 0 and iteration < self.maxIter:

            dernierCentres = self.centres.copy()
            iteration += 1

            distances = np.ones(
                (1, self.dataNums)) * np.sum((self.data - self.centres[0, :])**2, axis=1)

            for i in range(self.k - 1):
                distances = np.append(distances, np.ones(
                    (1, self.dataNums)) * np.sum((self.data - self.centres[i + 1, :])**2, axis=1), axis=0)

            cluster = np.argmin(distances, axis=0)
            cluster = np.transpose(cluster * np.ones((1, self.dataNums)))

            for i in range(self.k):
                currentCluster = np.where(cluster == i)[0]
                if currentCluster.shape[0] > 0:
                    self.centres[i, :] = np.sum(
                        self.data[currentCluster], axis=0) / currentCluster.shape[0]

            distances = np.ones(
                (1, self.dataNums)) * np.sum((self.data - self.centres[0, :])**2, axis=1)

            for i in range(self.k - 1):
                distances = np.append(distances, np.ones(
                    (1, self.dataNums)) * np.sum((self.data - self.centres[i + 1, :])**2, axis=1), axis=0)

            return self


    def transform(self, data):
        nData = np.shape(data)[0]

        distances = np.ones(
            (1, nData)) * np.sum((data - self.centres[0, :])**2, axis=1)
        for j in range(self.k - 1):
            distances = np.append(distances, np.ones(
                (1, nData)) * np.sum((data - self.centres[j + 1, :])**2, axis=1), axis=0)

        cluster = np.argmin(distances, axis=0)
        cluster = np.transpose(cluster * np.ones((1, nData)))
        new_inputs = np.concatenate((data, cluster), axis=1)

        return new_inputs

    def predict(self, data):

        nData = np.shape(data)[0]

        # On calcul la distance avec les centres
        distances = np.ones(
            (1, nData)) * np.sum((data - self.centres[0, :])**2, axis=1)
        for j in range(self.k - 1):
            distances = np.append(distances, np.ones(
                (1, nData)) * np.sum((data - self.centres[j + 1, :])**2, axis=1), axis=0)

        # On retourne le cluster
        cluster = distances.argmin(axis=0)

        return cluster
