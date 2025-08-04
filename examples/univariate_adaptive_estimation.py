from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import adaptive_periodogram, spectral_w_mask
from class_and_func.estimator_class import univariate_spectral_noised_estimator
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time


sns.set_theme()


def job(it, periodo, max_time, fixed_parameter):
    np.random.seed(it+100)

    estimator = univariate_spectral_noised_estimator(fixed_parameter)
    res = estimator.fit(periodo, max_time).x

    return res


if __name__ == "__main__":
    # Parameters and hyperparameters
    mu = np.array([[1/16]])
    alpha = np.array([[1/8]])
    beta = np.array([[30]])

    noise = np.array([[0.1]])
    parameters = np.array([mu, alpha, beta, noise])

    burn_in = -100
    max_time = 5000

    bounds = np.array([(1e-12, None), (1e-12, 1 - 1e-12), (1e-12, None), (1e-12, None)])

    # Fixed parameter (assumed known)
    idx_parameter = 0  # 0 = mu, 1 = alpha, 2 = beta, 3 = noise
    theta = parameters[[k for k in range(4) if k != idx_parameter]]

    # Simulation and construction of noised observations
    np.random.seed(0)

    hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
    hp.simulate()
    hp_times = hp.timestamps
    # print("lenH", len(hp_times), max_time * mu/(1-alpha))

    pp = multivariate_exponential_hawkes(noise, np.zeros((1,1)), np.ones((1,1)), max_time=max_time,
                                         burn_in=burn_in)
    pp.simulate()
    pp_times = pp.timestamps
    # print("lenP", len(pp_times), max_time*noise)

    idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
    parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

    # Estimation
    adapted_periodogram = adaptive_periodogram([parasited_times], max_time)
    adapted_K = len(adapted_periodogram)

    print("Adapted_K:", adapted_K)

    x_freq = np.arange(1, adapted_K + 1) / max_time

    f_x = np.array([spectral_w_mask((mu, alpha, beta, noise), x_0)[0] for x_0 in x_freq])

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 8))

    ax.plot(x_freq, f_x, label="Spectral density")

    ax.legend()

    print("Estimating...")
    start_time = time.time()
    res = job(0, adapted_periodogram, max_time, (idx_parameter, parameters[idx_parameter]))
    end_time = time.time()

    print("Total time of computation: " + str(end_time - start_time) + " sec.", end="\n\n")
    print("Real values:      ", theta.squeeze())
    print("Estimated values: ", res)

    plt.show()
