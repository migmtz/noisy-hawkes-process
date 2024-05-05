from class_and_func.multivariate_exponential_process import exp_thinning_hawkes
from class_and_func.spectral_functions import bartlett_periodogram, spectral_log_likelihood_grad_precomputed
import numpy as np
from scipy.optimize import minimize
import time


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

    hp = exp_thinning_hawkes(mu, alpha * beta, beta, t=burn_in, max_time=max_time)
    hp.simulate()
    ppp = exp_thinning_hawkes(noise, 0, beta, t=burn_in, max_time=max_time)
    ppp.simulate()

    times_hp = [0.0] + [t for t in hp.timestamps if 0 < t < max_time] + [max_time]
    times_pp = [t for t in ppp.timestamps if 0 < t < max_time]

    parasited_times = np.sort(times_hp + times_pp).tolist() # Should include bounds of window

    # Estimation
    M = int(len(parasited_times) - 2)

    periodogram = np.array([bartlett_periodogram(2 * np.pi * j / max_time, parasited_times) for j in range(1, M + 1)])

    print("Estimating...")
    start_time = time.time()
    res = minimize(spectral_log_likelihood_grad_precomputed,
                   (0.4, 0.4, 0.4),
                   method="L-BFGS-B", jac=True,
                   args=(M, periodogram, max_time, idx_parameter, parameters[idx_parameter]),
                   bounds=bounds[[k for k in range(4) if k != idx_parameter]], options=None)
    end_time = time.time()

    print("Total time of computation: " + str(end_time - start_time) + " sec.", end="\n\n")
    print("Real values:      ", theta)
    print("Estimated values: ", res.x)