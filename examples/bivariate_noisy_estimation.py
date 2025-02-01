from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import fast_multi_periodogram
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
import numpy as np
import time


def job(it, periodo, max_time):
    np.random.seed(it)
    mask = np.array([[True, False],
                     [True, True]])
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    return res.x


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

    np.random.seed(0)
    hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time)
    hp.simulate()
    hp_times = hp.timestamps

    pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time)
    pp.simulate()
    pp_times = pp.timestamps

    idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
    parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
    K = int(parasited_times.shape[0])

    periodogram = fast_multi_periodogram(K, parasited_times, max_time)

    start_time = time.time()
    res = job(0, periodogram, max_time)
    end_time = time.time()

    print(res)

    mu_est = np.array(res[0:2]).reshape((2,1))
    alpha_est = np.array([[res[2], 0], res[3:5]])

    print("Total time of computation: " + str(end_time - start_time) + " sec.", end="\n\n")
    print("         Real values:     Estimated values:")

    parameters_tri = ["mu_1    ", "mu_2    ", "alpha_11", "alpha_21", "alpha_22", "beta_1  ", "beta_2  ", "lambda_0"]
    for a, b, c in zip(parameters_tri, theta, res):
        print(a + "{0:8}            {1:8.3f}".format(b, c))
