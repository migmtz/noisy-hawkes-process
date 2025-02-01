from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import spectral_w_mask, fast_multi_periodogram_window
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

sns.set_theme()


def func(x, a, b, c):
    return (a ** 2) / (b ** 2 + (2 * np.pi * x) ** 2) + c ** 2


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def estimate_spectral_density(parasited_times_list, max_time, max_freq=2.0, smooth_parameter=25, epsilon=0.01):
    K = int(np.ceil(max_freq * max_time))
    x_freq = np.arange(-K, K + 1) / max_time

    IT_x = np.mean([fast_multi_periodogram_window(parasited_times, max_time, max_freq).real.squeeze() for parasited_times in
                    parasited_times_list], axis=0)

    smooth_window = K // smooth_parameter
    smooth_rolling_IT = []
    general_weights = np.array([stats.binom.pmf(i, 2 * smooth_window, 0.5) for i in range(2 * smooth_window + 1)])
    for i in range(0, len(IT_x)):

        aux = IT_x[max(i - smooth_window, 0): min(i + smooth_window, len(IT_x)) + 1]
        if i < smooth_window:
            weights = general_weights[smooth_window - i:] / stats.binom.cdf(smooth_window + i, 2 * smooth_window, 0.5)
        elif i >= len(IT_x) - smooth_window - 1:
            weights = general_weights[(smooth_window + i - len(IT_x)) + 1:] / stats.binom.cdf((smooth_window - i + len(IT_x) - 1),
                                                                                      2 * smooth_window, 0.5)
            weights = np.flip(weights)
        else:
            weights = general_weights

        smooth_rolling_IT += [np.average(aux, weights=weights)]

    popt, _ = curve_fit(func, x_freq[K + 1:], smooth_rolling_IT[K + 1:])
    a, b, c = popt
    est_max_freq = np.abs(b) * np.sqrt(1 / epsilon - 1) / (2 * np.pi)

    final_K = int(np.ceil(est_max_freq * max_time))

    return x_freq, smooth_rolling_IT, popt, est_max_freq, final_K, IT_x[K+1:K+1+final_K]


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

    x_freq, smooth_periodogram, popt, est_max_freq, final_K, final_ITx = estimate_spectral_density(parasited_times_list, max_time)
    f_x = np.array([spectral_w_mask((mu, alpha, beta, noise), x_0)[0] for x_0 in x_freq])
    K = int((len(x_freq) - 1)/2)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 8))

    ax.plot(x_freq, f_x, label="Spectral density")
    ax.plot(x_freq, smooth_periodogram, label="Rolling weighted average smoothing", alpha=0.5, zorder=0)

    xaux = x_freq[K:]
    yaux = func(x_freq[K:], *popt)

    ax.plot(xaux, yaux, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print("Explicit estimated:", est_max_freq)
    ax.plot([est_max_freq, est_max_freq], ax.get_ylim(), label="Estimated maximal frequency")

    ax.legend()


    plt.show()
