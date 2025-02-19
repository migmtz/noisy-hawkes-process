from class_and_func.spectral_functions import *
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import univariate_spectral_noised_estimator
import time
import pickle


def f_1(n):
    # M = N(T)
    return int(n)


def f_2(n):
    # M = N(T) log N(T)
    return int(n * np.log(n))


def job(it, periodo, max_time, fixed_parameter):
    np.random.seed(it+100)

    estimator = univariate_spectral_noised_estimator(fixed_parameter, initial_guess=(0.4, 0.4, 0.4))
    res = estimator.fit(periodo, max_time)

    aux = np.concatenate((res.x, np.array([end_time-start_time])))

    return aux


if __name__ == "__main__":

    mu = 1.0
    alpha = 0.5
    beta = 1.0

    burn_in = -100

    horizons = [250, 500, 1000, 2000, 4000, 8000]
    max_time = horizons[-1]
    M_functions = [f_1, f_2]
    repetitions = 50
    noise_levels = mu/(1 - alpha) * np.array([0.2 * k for k in range(1, 11)])

    estimations = np.zeros((3, 4, len(M_functions), len(noise_levels), repetitions))
    loglike_real = np.zeros((4, len(M_functions), len(noise_levels), repetitions))
    loglike_estim = np.zeros((4, len(M_functions), len(noise_levels), repetitions))
    estimation_time = np.zeros((4, len(M_functions), len(noise_levels), repetitions))
    simulated_nb_points = np.zeros((4, len(M_functions), len(noise_levels), repetitions))

    for idx_noise, noise in enumerate(noise_levels):
        print("Noise level: " + str(noise))
        parameters = np.array([mu, alpha, beta, noise])
        for idx_repetitions in range(repetitions):
            np.random.seed(idx_repetitions+100)
            hp = multivariate_exponential_hawkes(np.array([[mu]]), np.array([[alpha]]), np.array([[beta]]), max_time=max_time, burn_in=burn_in)
            hp.simulate()

            pp = multivariate_exponential_hawkes(noise * np.ones((1, 1)), 0 * np.array([[alpha]]), np.array([[beta]]), max_time=max_time,
                                                 burn_in=burn_in)
            pp.simulate()

            for idx_horizon, horizon in enumerate(horizons):
                hp_times = [(t,m) for t,m in hp.timestamps if 0 < t < horizon]
                pp_times = [(t,m) for t,m in pp.timestamps if 0 < t < horizon]

                idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
                parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

                for idx_M, M_func in enumerate(M_functions):
                    M = M_func(len(parasited_times) - 2)
                    periodogram = fast_multi_periodogram(M, parasited_times, max_time)

                    for idx_parameter, fixed_parameter in enumerate(parameters):
                        theta = parameters[[k for k in range(4) if k != idx_parameter]]
                        start_time = time.time()
                        estimator = univariate_spectral_noised_estimator((idx_parameter, fixed_parameter))
                        res = estimator.fit(periodogram, max_time)
                        end_time = time.time()

                        estimations[:, idx_parameter, idx_M, idx_noise, idx_repetitions] = res.x
                        loglike_real[idx_parameter, idx_M, idx_noise, idx_repetitions] = spectral_log_likelihood_grad_precomputed(theta.squeeze(), M, periodogram, horizon, idx_parameter, np.array([fixed_parameter]))[0]
                        loglike_estim[idx_parameter, idx_M, idx_noise, idx_repetitions] = res.fun
                        estimation_time[idx_parameter, idx_M, idx_noise, idx_repetitions] = end_time - start_time
                        simulated_nb_points[idx_parameter, idx_M, idx_noise, idx_repetitions] = len(parasited_times) - 2

            print("    Repetitions # " + str(idx_repetitions+1) + " of " + str(repetitions))
        print("Total time of computation: " + str(np.sum(estimation_time)) + " sec.", end="\n\n")

        toSave = [noise, estimations, loglike_real, loglike_estim, estimation_time, simulated_nb_points]

        #with open("saved_estimations_univariate/univariate_estimations", 'wb') as f:
        #    pickle.dump(toSave, f)