import numpy as np

from sklearn.preprocessing import LabelBinarizer

class mPerceptron:

    def __init__(self, lRate, iterationsNums):
        self.lRate = lRate
        self.iterationsNums = iterationsNums


    def fit(self, inputs, targets):

        self.inputs = inputs
        self.targets = LabelBinarizer().fit_transform(targets)

        # Nombre de perceptrons
        if np.ndim(inputs) > 1:
            self.inNums = np.shape(inputs)[1]
        else:
            self.inNums = 1

        if np.ndim(self.targets) > 1:
            self.outNums = np.shape(self.targets)[1]
        else:
            self.outNums = 1

        self.dataNums = np.shape(inputs)[0]

        # Initialisation des poids
        self.weights = np.random.rand(
            self.inNums + 1, self.outNums) * 0.1 - 0.05

        # Normalisation (Ajout du biais)
        self.inputs = np.concatenate(
            (inputs, -np.ones((self.dataNums, 1))), axis=1)

        # Copy of train function :
        for i in range(self.iterationsNums + 1):
            activations = self.activationFun()
            self.weights -= self.lRate * \
                np.dot(np.transpose(self.inputs), activations - self.targets)

    def activationFun(self):
        self.activations = np.dot(self.inputs, self.weights)
        return np.where(self.activations > 0, 1, 0)

    def train(self, lRate, iterationsNums):

        for i in range(iterationsNums + 1):
            activations = self.activationFun()
            self.weights -= lRate * \
                np.dot(np.transpose(self.inputs), activations - self.targets)

    def test(self, inputs, targets):
        # Normalisation pour inclure le calcul du biais
        inputs = np.concatenate(
            (inputs, np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(inputs, self.weights)
        classesNums = np.shape(targets)[1]

        if(classesNums == 1):
            classesNums = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cMatrix = np.zeros((classesNums, classesNums))

        for i in range(classesNums):
            for j in range(classesNums):
                cMatrix[i, j] = np.sum(
                    np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        precision = np.trace(cMatrix) / np.sum(cMatrix)

        return (cMatrix, precision)

    def predict(self, inputs):

        inputs = np.concatenate(
            (inputs, np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(inputs, self.weights)
        classesNums = np.shape(self.targets)[1]

        if(classesNums == 1):
            classesNums = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            outputs = np.argmax(outputs, 1)

        return outputs

    def predict_proba(self, inputs):
        inputs = np.concatenate(
            (inputs, np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(inputs, self.weights)
        outputs = 1/(1+np.exp(-outputs))
        return outputs
