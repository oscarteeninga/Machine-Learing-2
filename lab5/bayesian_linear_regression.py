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
        # Update the inverse covariance matrix (
        # Equation 77
        # Update the mean vector
        # Equation 78
        #
        return self

    def predict(self, x):
        # Obtain the predictive mean
        # Equation 62, Equation 80
        # Obtain the predictive variance
        # Equation 81
        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)

    @property
    def weights_dist(self):
        return stats.multivariate_normal(mean=self.mean, cov=self.cov)