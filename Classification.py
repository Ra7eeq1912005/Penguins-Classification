import numpy as np
import pandas as pd

# =============================== Perceptron Class ===============================
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100, add_bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.add_bias = add_bias
        self.weights = np.random.randn(input_size + (1 if add_bias else 0)) * 0.001
        self.errors = []

    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def train(self, X, y):
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]

        for epoch in range(self.epochs):
            epoch_errors = 0
            for i in range(len(X)):
                xi = X[i]
                target = y[i]
                v = np.dot(xi, self.weights)
                y_pred = self.activation(v)
                error = target - y_pred

                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    epoch_errors += 1

            self.errors.append(epoch_errors)
            if epoch_errors == 0:
                break

    def predict(self, X):
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        return self.activation(np.dot(X, self.weights))

    def get_boundary(self, x_vals):
        if len(self.weights) == 3:
            w0, w1, w2 = self.weights
            return (-w0 - w1 * x_vals) / w2
        elif len(self.weights) == 2:
            w1, w2 = self.weights
            return (-w1 * x_vals) / w2
        else:
            return None


# =============================== Adaline Class ===============================
class Adaline:
    def __init__(self, eta=0.01, epochs=100, mse_threshold=1e-3, add_bias=True):
        self.eta = eta
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias

    def fits(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.random.randn(n_features) * 0.001
        self.bias = np.random.randn() * 0.001 if self.add_bias else 0
        self.errors = []

        for epoch in range(self.epochs):
            for i in range(n_samples):
                x = X[i]
                d = y[i]
                v = np.dot(self.weight, x) + self.bias
                y_actual = v
                error = d - y_actual

                self.weight += self.eta * error * x
                if self.add_bias:
                    self.bias += self.eta * error

            net_input = np.dot(X, self.weight) + self.bias
            errors = y - net_input
            mse = (errors ** 2).mean() / 2
            self.errors.append(mse)

            if mse < self.mse_threshold:
                break
        return self

    def predict(self, X):
        net_input = np.dot(X, self.weight) + self.bias
        return np.where(net_input >= 0, 1, -1)

    def get_boundary(self, x_vals):
        if len(self.weight) == 2:
            w1, w2 = self.weight
            return (-self.bias - w1 * x_vals) / w2
        return None


 
def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    accuracy = (tp + tn) / len(y_true)
    return np.array([[tp, fn], [fp, tn]]), accuracy


def split_train_test_per_class(X, y, train_per_class=30, test_per_class=20):
    classes = np.unique(y)
    X_train, X_test, y_train, y_test = [], [], [], []

    for cls in classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        indices = np.arange(len(X_cls))
        np.random.shuffle(indices)

        X_train.append(X_cls[indices[:train_per_class]])
        X_test.append(X_cls[indices[train_per_class:train_per_class + test_per_class]])
        y_train.append(y_cls[indices[:train_per_class]])
        y_test.append(y_cls[indices[train_per_class:train_per_class + test_per_class]])

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

     
    train_idx = np.arange(len(X_train))
    test_idx = np.arange(len(X_test))
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]

     
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    return X_train, y_train, X_test, y_test
