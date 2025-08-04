from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import spectral_w_mask, fast_multi_periodogram_window
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from class_and_func.spectral_functions import func, smooth_periodogram

sns.set_theme()


if __name__ == "__main__":

    np.random.seed(2)
    mu = np.array([[1/8]])
    alpha = np.array([[1/8]])
    beta = np.array([[1.0]])

    noise = 0.2

    print(mu, alpha, beta, noise)

    avg_intensity = noise + mu / (1 - alpha)
    print("Average intensity", avg_intensity)

    max_time = 8000.0
    burn_in = -100

    repet = 1
    parasited_times_list = []
    for i in range(repet):
        np.random.seed(i)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, burn_in=burn_in, max_time=max_time)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(np.array([[noise]]), 0 * alpha, beta, burn_in=burn_in, max_time=max_time)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times_list += [np.array(pp_times[1:-1] + hp_times)[idx]]

    x_freq, smooth_periodogram, popt, est_max_freq, final_K, final_ITx = smooth_periodogram(parasited_times_list, max_time)
    f_x = np.array([spectral_w_mask((mu, alpha, beta, noise), x_0)[0] for x_0 in x_freq])
    K = int((len(x_freq) - 1)/2)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 8))

    ax.plot(x_freq, f_x, label="Spectral density")
    ax.plot(x_freq, smooth_periodogram, label="Rolling weighted average smoothing", alpha=0.5, zorder=0)

    xaux = x_freq[K:]
    yaux = func(x_freq[K:], *popt)

    ax.plot(xaux, yaux, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print("Explicit estimated maximal frequency:", est_max_freq)
    ax.plot([est_max_freq, est_max_freq], ax.get_ylim(), label="Estimated maximal frequency")

    ax.legend()


    plt.show()
