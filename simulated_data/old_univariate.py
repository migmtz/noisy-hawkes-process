from class_and_func.spectral_functions import *
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from scipy.optimize import minimize
import time
import pickle


def f_1(n):
    # M = N(T)
    return int(n)


def f_2(n):
    # M = N(T) log N(T)
    return int(n * np.log(n))
#TO CHECK

if __name__ == "__main__":

    mu = 1.0
    alpha = 0.5
    beta = 1.0

    burn_in = -100

    horizons = [250, 500, 1000, 2000, 4000, 8000]
    M_functions = [f_1]
    # repetitions = 50
    repetitions = 1
    #noise_levels = mu/(1 - alpha) * np.array([0.2 * k for k in range(1, 11)])
    noise_levels = [0.2 * i for i in range(1, 11)]

    bounds = np.array([(1e-12, None), (1e-12, 1 - 1e-12), (1e-12, None), (1e-12, None)])

    estimations = np.zeros((3, 4, len(M_functions), len(noise_levels), repetitions))
    loglike_real = np.zeros((4, len(M_functions), len(noise_levels), repetitions))
    loglike_estim = np.zeros((4, len(M_functions), len(noise_levels), repetitions))
    estimation_time = np.zeros((4, len(M_functions), len(noise_levels), repetitions))
    simulated_nb_points = np.zeros((4, len(M_functions), len(noise_levels), repetitions))

    for idx_noise, noise in enumerate(noise_levels):
        print("Noise level: " + str(noise))
        parameters = np.array([mu_levels[idx_noise], alpha, beta, noise])
        for idx_repetitions in range(repetitions):
            np.random.seed(idx_repetitions)
            hp = multivariate_exponential_hawkes(np.array([[mu_levels[idx_noise]]]), np.array([[alpha]]), np.array([[beta]]), max_time=max_time, burn_in=burn_in)
            hp.simulate()
            hp_times = hp.timestamps

            pp = multivariate_exponential_hawkes(noise * np.ones((1, 1)), 0 * np.array([[alpha]]), np.array([[beta]]), max_time=max_time,
                                                 burn_in=burn_in)
            pp.simulate()
            pp_times = pp.timestamps

            idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
            parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

            for idx_M, M_func in enumerate(M_functions):
                M = M_func(len(parasited_times) - 2)
                periodogram = fast_multi_periodogram(M, parasited_times, max_time)
                print(len(periodogram),M)

                for idx_parameter, fixed_parameter in enumerate(parameters):
                    if idx_parameter == 2:
                        theta = parameters[[k for k in range(4) if k != idx_parameter]]
                        start_time = time.time()
                        res = minimize(spectral_log_likelihood_grad_precomputed,
                                       (0.5, 0.5, 0.5),
                                       method="L-BFGS-B", jac=True,
                                       args=(M, periodogram, max_time, idx_parameter, fixed_parameter),
                                       bounds=bounds[[k for k in range(4) if k != idx_parameter]], options=None)
                        end_time = time.time()

                        estimations[:, idx_parameter, idx_M, idx_noise, idx_repetitions] = res.x
                        #loglike_real[idx_parameter, idx_M, idx_noise, idx_repetitions] = spectral_log_likelihood_grad_precomputed(theta, M, periodogram, horizon, idx_parameter, fixed_parameter)[0]
                        #loglike_estim[idx_parameter, idx_M, idx_noise, idx_repetitions] = res.fun
                        estimation_time[idx_parameter, idx_M, idx_noise, idx_repetitions] = end_time - start_time
                        simulated_nb_points[idx_parameter, idx_M, idx_noise, idx_repetitions] = len(parasited_times) - 2
                        #print(res.x, theta)
                        print(theta, res.x)

            print("    Repetitions # " + str(idx_repetitions+1) + " of " + str(repetitions))
        print("Total time of computation: " + str(np.sum(estimation_time)) + " sec.", end="\n\n")

        # toSave = [noise, estimations, loglike_real, loglike_estim, estimation_time, simulated_nb_points]

        # with open("saved_estimations_noise", 'wb') as f:
            # pickle.dump(toSave, f)