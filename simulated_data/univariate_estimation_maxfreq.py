import numpy as np
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import univariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram_window
from scipy import stats
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import time


def job(it, periodo, max_time, fixed_parameter):
    np.random.seed(it+100)

    estimator = univariate_spectral_noised_estimator(fixed_parameter)
    start_time = time.time()
    res = estimator.fit(periodo, max_time)
    end_time = time.time()

    aux = np.concatenate((res.x, np.array([end_time-start_time])))
    print('-', end='')

    return aux


def func(x, a, b, c):
    return (a ** 2) / (b ** 2 + (2 * np.pi * x) ** 2) + c ** 2


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

    mu = np.array([[1/8]])# 1.0 or 1/8
    alpha = np.array([[1/8]])# 0.5 or 1/8
    beta = np.array([[1.0]])

    noise_list = mu[0]/(1 - alpha[0]) * np.array([0.2 * k for k in range(1, 11)])
    print(noise_list)

    burn_in = -100
    repetitions = 50

    horizons = [250, 500, 1000, 2000, 4000, 8000]
    max_time = horizons[-1]

    fixed_list = ["mu", "alpha", "beta", "noise"]

    for noise in noise_list:

        theta = np.concatenate((mu, alpha, beta, np.array([[noise]])))
        fixed_parameter_list = [(i, theta[i]) for i in range(4)]

        # Simulations and periodograms

        print("parameters: ", mu,alpha,beta,noise)
        simulated_points = []
        start_time = time.time()
        for it in range(repetitions):
            np.random.seed(it)
            hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
            hp.simulate()
            hp_times = hp.timestamps

            pp = multivariate_exponential_hawkes(noise * np.ones((1, 1)), 0 * alpha, beta, max_time=max_time,
                                                 burn_in=burn_in)
            pp.simulate()
            pp_times = pp.timestamps

            idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
            parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
            simulated_points += [parasited_times]

        for horizon in horizons:
            periodogram_list = []
            start_time = time.time()
            for it in range(repetitions):
                aux_parasited = simulated_points[it][simulated_points[it][:, 0] <= horizon, :]
                _, _, _, _, final_K, periodogram = estimate_spectral_density([aux_parasited], max_time)
                print(final_K, end=" ")

                periodogram_list += [periodogram[:, np.newaxis, np.newaxis]]
            end_time = time.time()
            print("Periodogram time:", end_time - start_time)
            start_time = time.time()

            for j, fixed_parameter in enumerate(fixed_parameter_list):
                theta_aux = theta[[i != j for i in range(4)], 0]

                print('|' + '-' * (repetitions) + '|')
                print('|', end='')
                with Pool(cpu_count() - 3) as p:
                    estimations = np.array(
                        p.starmap(job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions),[fixed_parameter] * (repetitions))))
                print('|\n Done')

                print("*"*70)

                np.savetxt("saved_estimations_univariate/horizons_revision/" + str(horizon) + "univariate_horizon_2adaptive_"+ fixed_list[j] + "_" + str(np.round(noise, 2)) + ".csv", estimations,
                           delimiter=",")
            end_time = time.time()
            print("Estimation time:", end_time - start_time)
            print("*" * 100)
            print("*" * 100)

