import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from bayesian_linear_regression import BayesianLinearRegression


def generate_points_with_noise(size):
    pure = np.linspace(-1, 1, size)
    noise = np.random.normal(0, 0.2, size)
    return [[pure[p], -0.2 + 0.6 * pure[p] + noise[p]] for p in range(len(pure))]


def test_dataset():
    print(generate_points_with_noise(10))


def test_boston():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random.randint(0, 100))
    model = BayesianLinearRegression(n_features=len(np.unique(y)), alpha=0.3, beta=1)
    for i in range(len(X_train)):
        model.predict()
