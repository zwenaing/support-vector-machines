import numpy as np
from cvxopt import matrix, solvers

class SVM:

    def __init__(self, data_X, data_y, kernel='dot_product'):
        self.data_X = data_X
        self.data_y = data_y
        self.kernel = kernel

    def dual_form_optimization(self, C, kernel):
        """
        This is the procedure for solving the dual form of SVM,
        including P, q, G, h, A and b formulations and cvxopt library call.
        :param C: C - slack variables penalty
        :param kernel: kernel function, default to dot product
        :return: optimized alphas
        """
        n = self.data_X.shape[0]  # number of rows of X

        P = matrix(np.dot(self.data_y, np.transpose(self.data_y)) * kernel(self.data_X))  # shape: n x n
        # [[-1]
        #  [-1]
        #  [..]
        #  [-1]]
        q = matrix(-1 * np.ones((n, 1)))  # shape: n x 1

        # [[ dig(n) with -1]
        #  [    I (nxn)    ]]
        G_no_slack = matrix(-1 * np.eye(n))
        G_slack = matrix(np.eye((n)))
        G = matrix(np.vstack((G_no_slack, G_slack)))  # shape: 2n x n

        # [[0]
        #  [0]
        #  ...
        #  [0]
        #  [C]
        #  [C]
        #  ...
        #  [C]]
        h_no_slack = matrix(np.zeros(n))
        h_slack = matrix(np.ones(n) * C)
        h = matrix(np.vstack((h_no_slack, h_slack)))  # shape: 2n x 1

        A = matrix(self.data_y.reshape(1, -1))  # shape: 1 x n
        b = matrix(np.zeros(1))  # shape: 1 x n

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        return alphas

    def fit(self, C, kernel="dot", sigma=None, degree=None, constant=None):
        """
        Fit the SVM to the data and return the weights and bias term.
        :param C: the slack variable penalty.
        :param kernel: the kernel function. detault to dot product else use rbf for gaussian kernel
                        and poly for polynomial kernel.
        :param sigma: the sigma parameter for gaussian kernel.
        :param degree: the degree of the polynomial.
        :param constant: the constant of the polynomial.
        :return: the weights and bias term.
        """
        if kernel == "rbf":
            kernel = self.rbf(sigma)
        elif kernel == "poly":
            kernel = self.svm_polynomial_kernel(degree, constant)
        else:
            kernel = lambda X: np.dot(X, np.transpose(X))
        alphas = self.dual_form_optimization(C, kernel)  # get alphas
        w = np.sum(alphas * self.data_y * self.data_X[:, 1:], axis=0)
        cond = (alphas > 1e-7).reshape(-1)
        b = self.data_y[cond] - np.dot(self.data_X[cond][:, 1:], w).reshape(-1, 1)
        if b.size == 0:
            raise Exception("All b values , threshold. Try decreasing the threshold.")
        else:
            bias = b[0]
        return w, bias

    def svm_polynomial_kernel(self, power, constant=0):
        """
        Return the function for computing polynomial kernel.
        :param power: the power of polynomial kernel.
        :param constant: the constant of polynomial kernel.
        :return: the polynomial kernel.
        """
        return lambda X: np.power(np.dot(X, np.transpose(X)) + constant, power)

    def gaussian_kernel(self, x, z, sigma):
        """
        Compute gaussian kernel between two data point.
        :param x: the first data point.
        :param z: the second data point.
        :param sigma: the gaussian parameter.
        :return: the gaussian kernel result.
        """
        sub = x - z
        res = np.exp(np.dot(np.transpose(sub), sub) / - 2 * sigma ** 2)
        return res

    def rbf(self, sigma):
        """
        Return the function for computing rbf kernel.
        :param sigma: the rbf kernel parameter.
        :return: the rbf kernel.
        """
        n = self.data_X.shape[0]
        kernel_matrix = np.zeros((n, n))
        for i, x in enumerate(self.data_X):
            for j, z in enumerate(self.data_X):
                kernel_matrix[i, j] = self.gaussian_kernel(x, z, sigma)
        return kernel_matrix

