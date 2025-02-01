import numpy as np
from multiprocessing import Pool, cpu_count
import time
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram


def job(it, periodo, max_time):
    np.random.seed(it)
    estimator = multivariate_spectral_noised_estimator()
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


def reduced_job(it, periodo, max_time):
    np.random.seed(it)
    mask = np.array([[False, False],
                     [True, False]])
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


if __name__ == "__main__":
    # Parameters
    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.0, 0.0],
                      [0.4, 0.0]])
    beta = np.array([[1.0],
                     [1.3]])
    noise = 0.5

    print("Spectral radius:", np.max(np.abs(np.linalg.eig(alpha)[0])))

    max_time = 3000
    burn_in = -100
    repetitions = 50

    K_func = lambda x : int(x)

    # Simulations and periodograms

    periodogram_list = []

    start_time = time.time()
    for it in range(repetitions):
        np.random.seed(it)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time,  burn_in=burn_in)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

        K = K_func(parasited_times.shape[0])
        periodogram = fast_multi_periodogram(K, parasited_times, max_time)

        periodogram_list += [periodogram]

    end_time = time.time()
    print("Periodogram time:", end_time - start_time)

    print("Estimation in full model:")
    print('|' + '-' * (repetitions) + '|')
    print('|', end='')
    start_time = time.time()
    with Pool(5) as p:
        estimations = np.array(p.starmap(job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions))))
    print('|\n Done')
    # estimations = np.array([job(it, periodo) for it, periodo in zip(range(repetitions), periodogram_list)])
    end_time = time.time()
    print("Estimation time:", end_time - start_time)

    np.savetxt("saved_estimations/multivariate_single_" + str(max_time) + ".csv", estimations, delimiter=",")

    print("")
    print("Estimation in reduced model:")
    print('|' + '-' * (repetitions) + '|')
    print('|', end='')
    start_time = time.time()
    with Pool(5) as p:
        reduced_estimations = np.array(
            p.starmap(reduced_job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions))))
    print('|\n Done')
    # estimations = np.array([job(it, periodo) for it, periodo in zip(range(repetitions), periodogram_list)])
    end_time = time.time()
    print("Estimation time:", end_time - start_time)

    np.savetxt("saved_estimations/multivariate_single_red_" + str(max_time) + ".csv", reduced_estimations,
               delimiter=",")
