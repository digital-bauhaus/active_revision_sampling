#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import GPy as gpy


def mape(truth, prediction):
    return np.mean( np.abs((truth - prediction) / truth) )

class GPRegressor:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys= ys

    def fit(self):
        pass

    def predict(self):
        pass

class GPRegressor_gpy(GPRegressor):
    def __init__(self, xs, ys, kernel=gpy.kern.Brownian(input_dim=1)):
        self.xs = xs
        self.ys = ys
        self.kernel = kernel

    def fit(self, training_sample):
        self.training_sample = training_sample
        #print(training_sample)
        xs = self.xs.iloc[training_sample].values.reshape(-1, 1)
        ys = self.ys.iloc[training_sample].values.reshape(-1, 1)
        self.model = gpy.models.GPRegression(
            xs,
            ys,
            kernel=self.kernel
        )

        self.model.constrain_positive('')
        self.model.optimize()

    def predict(self):
        mean, std = self.model.predict(self.xs.values.reshape(-1, 1))
        return mean, std

class ActiveLearner:

    def __init__(self, xs, ys, kernel):
        ys = ys - ys.iloc[0]
        self.gp = GPRegressor_gpy(xs, ys, kernel=kernel)

    def prepare(self, size = 3):
        """
        Randomly select an initial training set of size size.

        :param size: size of initial training set
        :return:
        """
        self.training = np.array([0, self.gp.xs.shape[0]//2, self.gp.xs.shape[0] - 1])

    def train(self, randomized = False, size = 10):
        """
        Wrapper for self.gp.fit()
        :return:
        """
        self.gp.fit(self.training)

    def acquire_next(self, method = "uncertainty"):
        """
        Acquires the next measurement to add to the training set based on heuristics for
        expected improvement of the regression model.

        :param method: either 'uncertainty' or 'bisect'
        :return:
        """

        #xs = self.gp.xs.iloc[self.training].values
        #ys = self.gp.ys.iloc[self.training].values.reshape(1, -1)[0]

        def acquire_next_by_uncertainty():
            mean, std = self.gp.predict()
            next = np.argmax(std)
            if next not in self.training:
                #print(next, " not in ", self.training)
                self.training = np.append(self.training, next)
                self.next = next
            else:
                while next in self.training:
                    pool = set(list(range(len(self.gp.xs)))).difference(set(self.training))
                    next = random.choice(list(pool))
                self.next = next
                self.training = np.append(self.training, next)

        def acquire_next_random():
            next_sample = 0
            while next in self.training:
                pool = set(list(range(len(self.gp.xs)))).difference(set(self.training))
                next_sample = random.choice(list(pool))
            self.next = next
            self.training = np.append(self.training, next_sample)

        if method == "uncertainty":
            acquire_next_by_uncertainty()
        elif method == "random":
            acquire_next_random()
        else:
            raise ValueError("")

    def validate(self):
        """
        Compares the prediction of the ActiveLearner to the actual performance
        measurements providing the MAPE error metric

        :return: (mean absolute percentage error)
        """

        prediction_mean, prediction_std = self.gp.predict()
        return mape(prediction_mean, self.ys)



