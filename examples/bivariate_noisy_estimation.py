from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.spectral_functions import multivariate_periodogram, spectral_multivariate_noised_ll_trinf
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import time


if __name__ == "__main__":
    # Parameters
    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.5, 0.0],
                      [0.4, 0.4]])
    beta = np.array([[1.0],
                     [1.3]])

    print("Spectral radius:", np.max(np.abs(np.linalg.eig(alpha)[0])))

    max_time = 2500

    noise = 0.5
    theta = np.concatenate((mu.ravel(), alpha.ravel(), beta.ravel()))
    theta = np.append(theta[theta > 0.0], noise)

    bounds = [(1e-16, None), (1e-16, None), (1e-16, 1 - 1e-16), (1e-16, 1 - 1e-16), (1e-16, 1 - 1e-16),
              (1e-16, None), (1e-16, None), (1e-16, None)]

    np.random.seed(0)
    hp = multivariate_exponential_hawkes(mu, alpha * beta, beta, max_time=max_time)
    hp.simulate()
    hp_times = hp.timestamps

    pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time)
    pp.simulate()
    pp_times = pp.timestamps

    idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
    parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
    K = int(parasited_times.shape[0])
    init = np.random.rand(8) / 2 + np.r_[.75, .75, 0., 0., 0., .75, .75, .75]

    periodogram = [multivariate_periodogram(j / max_time, parasited_times) for j in range(1, K + 1)]

    start_time = time.time()
    res = minimize(spectral_multivariate_noised_ll_trinf,
                   init, tol=1e-16,
                   method="L-BFGS-B", jac=None,
                   args=(periodogram, K, max_time),
                   bounds=bounds, options={"disp": False})
    end_time = time.time()

    mu_est = np.array(res.x[0:2]).reshape((2,1))
    alpha_est = np.array([[res.x[2], 0], res.x[3:5]])

    print("Total time of computation: " + str(end_time - start_time) + " sec.", end="\n\n")
    print("         Real values:     Estimated values:")

    parameters_tri = ["mu_1    ", "mu_2    ", "alpha_11", "alpha_21", "alpha_22", "beta_1  ", "beta_2  ", "lambda_0"]
    for a, b, c in zip(parameters_tri, theta, res.x):
        print(a + "{0:8}            {1:8.3f}".format(b, c))
