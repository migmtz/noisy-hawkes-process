from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import adaptive_periodogram, spectral_w_mask
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time


sns.set_theme()


def job(it, periodo, max_time):
    np.random.seed(it)
    mask = np.array([[True, False],
                     [True, True]])
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    return res.x


titles = ["$f_{11}$", "$f_{22}$"]


if __name__ == "__main__":
    # Parameters and hyperparameters
    mu = np.array([[1/16],
                   [1.0]])
    alpha = np.array([[1/8, 0.0],
                      [0.3, 0.2]])
    beta = np.array([[30,
                      1.3]])

    noise = np.array([[0.1]])

    burn_in = -100
    max_time = 5000

    # Simulation and construction of noised observations
    np.random.seed(0)

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

    # Estimation
    adapted_periodogram = adaptive_periodogram([parasited_times], max_time)
    adapted_K = adapted_periodogram.shape[0]

    print("Adapted_K:", adapted_K)

    x_freq = np.arange(1, adapted_K + 1) / max_time

    theta_aux = (mu, alpha, beta, noise)
    f_x = np.array([spectral_w_mask(theta_aux, j / max_time) for j in range(1, adapted_K + 1)])

    fig, ax = plt.subplots(1, 2, sharex=False, figsize=(12, 4.5))

    for k in range(2):
        ax[k].plot(x_freq, f_x[:, k, k], label="Spectral density")
        ax[k].set_title(titles[k])
        ax[k].set_xlabel("$\\nu$")

    print("Estimating...")
    start_time = time.time()
    res = job(0, adapted_periodogram, max_time)
    end_time = time.time()

    print("Total time of computation: " + str(end_time - start_time) + " sec.", end="\n\n")
    print("         Real values:     Estimated values:")

    parameters_tri = ["mu_1    ", "mu_2    ", "alpha_11", "alpha_21", "alpha_22", "beta_1  ", "beta_2  ", "lambda_0"]
    for a, b, c in zip(parameters_tri, theta, res):
        print(a + "{0:8}            {1:8.3f}".format(b, c))

    plt.show()
