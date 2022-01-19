from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import random
from naive_bayes_classificator import naive_bayes, accuracy_metric
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def test_iris():
    X, y = load_iris(return_X_y=True)
    accuracy = 0
    for i in range(0, 20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random.randint(0, 100))
        accuracy += accuracy_metric(naive_bayes(X_train, y_train, X_test, y_test), y_test)
    print("\n====IRIS RESULTS====")
    print("Accuracy: " + str(accuracy/20))
    print("====================")


def test_wine_standard_scalar():

    tests = 100
    # Standard
    X, y = load_wine(return_X_y=True)
    accuracy = 0
    for i in range(0, tests):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random.randint(0, 100))
        accuracy += accuracy_metric(naive_bayes(X_train, y_train, X_test, y_test), y_test)
    print("\n====WINE RESULTS====")
    print("Accuracy: " + str(accuracy/tests))
    print("====================")

    # Standard Scalar
    X_scalar = StandardScaler().fit_transform(X)
    accuracy = 0
    for i in range(0, tests):
        X_train, X_test, y_train, y_test = train_test_split(X_scalar, y, test_size=0.3, random_state=random.randint(0, 100))
        accuracy += accuracy_metric(naive_bayes(X_train, y_train, X_test, y_test), y_test)
    print("\n====WINE SCALAR RESULTS====")
    print("Accuracy: " + str(accuracy/tests))
    print("====================")

    # PCA
    X_pca = PCA(n_components=2).fit_transform(X)
    accuracy = 0
    for i in range(0, tests):
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=random.randint(0, 100))
        accuracy += accuracy_metric(naive_bayes(X_train, y_train, X_test, y_test), y_test)
    print("\n====WINE PCA RESULTS====")
    print("Accuracy: " + str(accuracy/tests))
    print("====================")
