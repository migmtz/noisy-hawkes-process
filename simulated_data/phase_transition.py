import numpy as np
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram
from multiprocessing import Pool, cpu_count
import time
import pickle


def job(it, periodo, max_time):
    np.random.seed(it)
    mask = np.array([[False, False],
                     [True, False]])
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


if __name__ == "__main__":
    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.0, 0.0],
                      [0.2, 0.0]])
    beta = np.array([[1.0],
                     [1.3]])
    noise = 0.5
    dim = 2

    n_trials = 10

    the_max_time = [100, 1000, 3000, 5000, 10000]
    the_alpha0 = [0.2, 0.4, 0.6, 0.8]

    pairs = np.array(np.meshgrid(the_max_time, the_alpha0)).T.reshape(-1, 2)
    print(pairs)
    periodogram_list = []

    for trial in range(n_trials):
        aux_periodo = []
        for ijob, pair in enumerate(pairs):

            np.random.seed(ijob + 42 + trial * 100)
            max_time, alpha0 = pair

            # Store results
            result = dict(max_time=max_time, alpha=alpha0)

            # Set variates
            alpha[1, 0] = alpha0

            # Simulation
            hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time)
            hp.simulate()
            hp_times = hp.timestamps

            pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time)
            pp.simulate()
            pp_times = pp.timestamps

            idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
            parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
            K = int(parasited_times.shape[0])

            # Periodogram computation
            periodogram = fast_multi_periodogram(K, parasited_times, max_time)
            aux_periodo += [periodogram]
        periodogram_list += [aux_periodo]

    # Load previous results
    results = []

    # Progress bar
    n_jobs = n_trials * pairs.shape[0]
    print(n_jobs, pairs.shape[0])
    print('|' + '-' * n_jobs + '|')
    print("|", end="")
    for trial in range(n_trials):
        # Do the job
        with Pool(cpu_count() - 3) as p:
            res_pool = p.starmap(job, zip(range(pairs.shape[0]), periodogram_list[trial], pairs[:, 0]))

        # Add trial info
        #for res in res_pool:
        #    res['trial'] = trial
        results += res_pool
        #print(results)

        # Save
        with open('saved_estimations/phase_transition/results.pkl', 'wb') as fo:
            pickle.dump(results, fo)

    print('\n Done')