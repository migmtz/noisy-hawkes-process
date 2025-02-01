from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import fast_multi_periodogram
from class_and_func.estimator_class import univariate_spectral_noised_estimator
import numpy as np
from scipy.optimize import minimize
import time


def job(it, periodo, max_time, fixed_parameter):
    np.random.seed(it+100)

    estimator = univariate_spectral_noised_estimator(fixed_parameter)
    res = estimator.fit(periodo, max_time).x

    return res


if __name__ == "__main__":
    # Parameters and hyperparameters
    mu = 1.0
    alpha = 0.5
    beta = 0.3

    noise = 0.5
    parameters = np.array([mu, alpha, beta, noise])

    burn_in = -100
    max_time = 5000

    bounds = np.array([(1e-12, None), (1e-12, 1 - 1e-12), (1e-12, None), (1e-12, None)])

    # Fixed parameter (assumed known)
    idx_parameter = 0  # 0 = mu, 1 = alpha, 2 = beta, 3 = noise
    theta = parameters[[k for k in range(4) if k != idx_parameter]]

    # Simulation and construction of noised observations
    np.random.seed(0)

    hp = multivariate_exponential_hawkes(np.array([[mu]]), np.array([[alpha]]), np.array([[beta]]), max_time=max_time, burn_in=burn_in)
    hp.simulate()
    hp_times = hp.timestamps
    # print("lenH", len(hp_times), max_time * mu/(1-alpha))

    pp = multivariate_exponential_hawkes(noise * np.ones((1, 1)), np.zeros((1,1)), np.ones((1,1)), max_time=max_time,
                                         burn_in=burn_in)
    pp.simulate()
    pp_times = pp.timestamps
    # print("lenP", len(pp_times), max_time*noise)

    idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
    parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

    # Estimation
    K = int(len(parasited_times) - 2)

    periodogram = fast_multi_periodogram(K, parasited_times, max_time)

    print("Estimating...")
    start_time = time.time()
    res = job(0, periodogram, max_time, (idx_parameter, parameters[idx_parameter]))
    end_time = time.time()

    print("Total time of computation: " + str(end_time - start_time) + " sec.", end="\n\n")
    print("Real values:      ", theta)
    print("Estimated values: ", res)