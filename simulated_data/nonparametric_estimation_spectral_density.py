from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.spectral_functions import spectral_w_mask, fast_multi_periodogram_window
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


def func(x, a, b, c):
    return (a**2)/(b**2 + (2*np.pi*x)**2) + c**2


def estimate_spectral_density_smoothing(parasited_times_list, max_time, max_freq=2.0, smooth_parameter=25, epsilon=0.01):
    K = int(np.ceil(max_freq * max_time))
    x_freq = np.arange(-K, K + 1) / max_time

    IT_x = np.mean([fast_multi_periodogram_window(parasited_times, max_time, max_freq).real.squeeze() for parasited_times in
                    parasited_times_list], axis=0)

    smooth_window = K // smooth_parameter
    smooth_rolling_IT = []
    smooth_rolling_centered = []
    general_weights = np.array([stats.binom.pmf(i, 2 * smooth_window, 0.5) for i in range(2 * smooth_window + 1)])
    for i in range(0, len(IT_x)):

        aux = IT_x[max(i - smooth_window, 0): min(i + smooth_window, len(IT_x)) + 1]
        smooth_rolling_IT += [np.mean(aux)]

        if i < smooth_window:
            weights = general_weights[smooth_window - i:] / stats.binom.cdf(smooth_window + i, 2 * smooth_window, 0.5)
        elif i >= len(IT_x) - smooth_window - 1:
            weights = general_weights[(smooth_window + i - len(IT_x)) + 1:] / stats.binom.cdf((smooth_window - i + len(IT_x) - 1),
                                                                                      2 * smooth_window, 0.5)
            weights = np.flip(weights)
        else:
            weights = general_weights

        smooth_rolling_centered += [np.average(aux, weights=weights)]

    return x_freq, IT_x, smooth_rolling_IT, smooth_rolling_centered

if __name__ == "__main__":
    sns.set_theme()

    np.random.seed(2)
    mu = np.array([[1.0]])
    alpha = np.array([[0.5]])
    beta = np.array([[1.0]])

    print("Average intensity:", mu/(1-alpha))
    print("Bump at 0:", mu/((1-alpha)**3))

    noise = 1.0

    avg_intensity = noise + mu / (1 - alpha)
    print("avg_intensity", avg_intensity)

    max_time = 500.0
    burn_in = -100

    max_freq = 10.0
    nb_freq = int(np.ceil(max_freq * max_time))

    repetitions = 10
    parasited_times_list = []
    for i in range(repetitions):
        np.random.seed(i)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, burn_in=burn_in, max_time=max_time)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(np.array([[noise]]), 0 * alpha, beta, burn_in=burn_in, max_time=max_time)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
        parasited_times_list += [parasited_times]

    x_freq, IT_x, smooth_rolling_IT, smooth_rolling_centered = estimate_spectral_density_smoothing(parasited_times_list, max_time, max_freq=max_freq)

    f_x = np.array([spectral_w_mask((mu, alpha, beta, noise), x_0)[0] for x_0 in x_freq])

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(15,6))

    ax[0].plot(x_freq, f_x, label="Spectral density")
    ax[0].plot(x_freq, IT_x, c="r", alpha=0.5, label="Periodogram")

    ax[1].plot(x_freq, f_x, label="Spectral density")
    ax[1].plot(x_freq, smooth_rolling_IT, label="Rolling average smoothing")

    ax[2].plot(x_freq, f_x, label="Spectral density")
    ax[2].plot(x_freq, smooth_rolling_centered, label="Rolling weighted average smoothing")

    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")
    ax[2].legend(loc="upper left")

    plt.show()
