import numpy as np


class PCA(object):
    def __init__(self, d=2, sigma=1):
        self.d = d
        self.sigma = sigma

    @staticmethod
    def mean(x_data):
        mu = np.array(np.sum(x_data, axis=1)) / len(x_data[0, :])
        return np.expand_dims(mu, axis=0).T

    def pca(self, x_data):
        X = np.array(x_data)
        mean = self.mean(X)
        U, s, V = np.linalg.svd(X - mean, full_matrices=True)
        basis = U[:, 0:self.d]
        new_x_data = np.matmul(basis.T, X - mean)
        return mean, basis, new_x_data

