import scipy.stats as stats
import numpy as np


class BayesianLinearRegression:
    def __init__(self, n_features, alpha, beta):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.mean = np.zeros(n_features)
        self.cov = np.identity(n_features) * alpha

    def learn(self, x, y):
        cov_inv_prev = np.linalg.inv(self.cov)
        cov_inv = cov_inv_prev + self.beta * np.outer(x, x)
        cov = np.linalg.inv(cov_inv)
        mean = cov @ (cov_inv_prev @ self.mean + self.beta * y * x)
        self.cov = cov
        self.mean = mean
        return self

    def predict(self, x):
        y_pred_mean = x @ self.mean
        y_pred_var = 1 / self.beta + x @ self.cov @ x.T
        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)

    @property
    def weights_dist(self):
        return stats.multivariate_normal(mean=self.mean, cov=self.cov)
