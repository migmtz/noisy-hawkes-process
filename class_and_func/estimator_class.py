import numpy as np
from scipy.optimize import minimize
from class_and_func.spectral_functions import *


class univariate_spectral_noised_estimator(object):
    def __init__(self, fixed_parameter, loss=spectral_log_likelihood_grad_precomputed, grad=True, initial_guess="random", options=None):
        self.idx_param, self.fixed_parameter = fixed_parameter
        self.loss = loss
        self.grad = grad  # By default, uses grad version of spectral ll
        self.initial_guess = initial_guess

        if options is None:
            self.options = {'disp': False}#,"maxls":40}
        else:
            self.options = options

    def fit(self, periodogram, max_time):

        #np.random.seed(0)

        K = int(periodogram.shape[0])
        self.dim = (periodogram[0]).shape[0]

        # Bounds
        bounds = [(1e-16, None), (1e-16, 1 - 1e-16), (1e-16, None), (1e-16, None)]

        # Initial point
        if isinstance(self.initial_guess, str) and self.initial_guess == "random":
            init_a = np.random.uniform(0, 3, 3)
            init_alpha = np.random.uniform(0, 1, (self.dim, self.dim))

            init = np.concatenate((init_a[0].ravel(), init_alpha.ravel(), init_a[1:].ravel()))

        # Mask of non-fixed parameters
        indices = [i != self.idx_param for i in range(4)]
        bounds = np.array(bounds)[indices]
        init = init[indices]

        self.res = minimize(self.loss,
                            init, tol=1e-16,
                            method="L-BFGS-B", jac=self.grad,
                            args=(K, periodogram, max_time, self.idx_param,self.fixed_parameter.ravel()),
                            bounds=bounds, options=self.options)

        return self.res


class multivariate_spectral_unnoised_estimator(object):
    def __init__(self, loss=grad_ll_unnoised_mask, grad=True, initial_guess="random", mask=None, options=None):
        self.loss = loss
        self.grad = grad  # By default uses grad version of spectral ll
        self.initial_guess = initial_guess
        self.mask = mask

        if options is None:
            self.options = {'disp': False, "maxiter":1000}
        else:
            self.options = options

    def fit(self, periodogram, max_time):

        K = int(periodogram.shape[0])
        self.dim = (periodogram[0]).shape[0]

        # Bounds
        bounds = [(1e-16, None)] * self.dim
        bounds += ([(1e-16, 1 - 1e-16)] + [(1e-16, None)] * self.dim) * (self.dim - 1) + [(1e-16, 1 - 1e-16)]
        bounds += [(1e-16, None)] * (self.dim)

        # Initial point
        if isinstance(self.initial_guess, str) and self.initial_guess == "random":
            init_a = np.random.uniform(0, 3, self.dim * 2)

            a = np.random.uniform(0, 3, (self.dim, self.dim))
            radius = np.max(np.abs(np.linalg.eig(a)[0]))
            div = np.random.uniform(1e-16, 1 - 1e-16)
            init_alpha = a * div / (radius)

            self.init = np.concatenate((init_a[0:2].ravel(), init_alpha.ravel(), init_a[2:].ravel()))

        # Mask of parameters
        if self.mask is not None:
            param_mask = np.concatenate(([True] * self.dim, self.mask.ravel(), self.mask.any(axis=1)))
            bounds = np.array(bounds)[param_mask]
            self.init = self.init[param_mask]

        else:
            param_mask = np.array([True]*(self.dim * (2 + self.dim)))

        # Estimation
        self.res = minimize(self.loss,
                            self.init, tol=1e-8,
                            method="L-BFGS-B", jac=self.grad,
                            args=(periodogram, K, max_time, self.mask),
                            bounds=bounds, options=self.options)

        return self.res


class multivariate_spectral_noised_estimator(object):
    def __init__(self, loss=grad_ll_mask, grad=True, initial_guess="random", mask=None, options=None):
        self.loss = loss
        self.grad = grad  # By default uses grad version of spectral ll
        self.initial_guess = initial_guess
        self.mask = mask

        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, periodogram, max_time):

        K = int(periodogram.shape[0])
        self.dim = (periodogram[0]).shape[0]

        # Bounds
        bounds = [(1e-16, None)] * self.dim
        bounds += ([(1e-16, 1 - 1e-16)] + [(1e-16, None)] * self.dim) * (self.dim - 1) + [(1e-16, 1 - 1e-16)]
        bounds += [(1e-16, None)] * (self.dim + 1)

        # Initial point
        if isinstance(self.initial_guess, str) and self.initial_guess == "random":
            init_a = np.random.uniform(0, 3, self.dim * 2 + 1)

            a = np.random.uniform(0, 3, (self.dim, self.dim))
            radius = np.max(np.abs(np.linalg.eig(a)[0]))
            div = np.random.uniform(1e-16, 1 - 1e-16)
            init_alpha = a * div / (radius)

            self.init = np.concatenate((init_a[0:2].ravel(), init_alpha.ravel(), init_a[2:].ravel()))

        # Mask of parameters
        if self.mask is not None:
            param_mask = np.concatenate(([True] * self.dim, self.mask.ravel(), self.mask.any(axis=1), [True]))
            bounds = np.array(bounds)[param_mask]
            self.init = self.init[param_mask]

        else:
            param_mask = np.array([True]*(self.dim * (2 + self.dim) + 1))

        # Estimation
        self.res = minimize(self.loss,
                            self.init, tol=1e-16,
                            method="L-BFGS-B", jac=self.grad,
                            args=(periodogram, K, max_time, self.mask),
                            bounds=bounds, options=self.options)

        return self.res