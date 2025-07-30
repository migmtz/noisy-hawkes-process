import numpy as np
from multiprocessing import Pool, cpu_count
import time
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram, multivariate_spectral_noised_density
from matplotlib import pyplot as plt


def job(it, periodo, max_time):
    np.random.seed(it) #__, +1
    mask = np.array([[True, True],
                     [True, True]])
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return np.hstack((res.x, res.fun))


def reduced_job(it, periodo, max_time, mask):
    np.random.seed(it)
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return np.hstack((res.x, res.fun))

mask_list_id = [np.array([[True, False],
                       [True, False]]),
             np.array([[False, True],
                       [False, True]]),
             np.array([[True, False],
                       [True, True]]),
             np.array([[True, True],
                       [False, True]])]
mask_list_nonid = [np.array([[True, False],
                       [False, True]]),
             np.array([[True, True],
                       [False, False]]),
             np.array([[False, False],
                       [True, True]])
             ]



if __name__ == "__main__":
    dim = 2

    mu = np.array([[0.01],
                   [0.15]])
    alpha = np.array([[0.80, 0.00],
                      [3.5, 0.70]])
    beta = np.array([[15.0],
                     [35.0]])
    noise = 0.05
    dossier = "similar_parameters"
    max_time = 145

    real_theta = [mu, alpha, beta, np.array([noise])]

    print("Spectral radius:", np.max(np.abs(np.linalg.eig(alpha)[0])))

    burn_in = -100
    repetitions = 25

    K_func = lambda x : int(x * np.log(x))

    ####

    # Simulations and periodograms

    periodogram_list = []

    n = 0

    start_time = time.time()
    for it in range(repetitions):
        np.random.seed(it+50)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time,  burn_in=burn_in)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
        n += len(parasited_times)
        #print(hp.timestamps.shape)

        K = K_func(parasited_times.shape[0])
        #K = int(max_time * 10)
        periodogram = fast_multi_periodogram(K, parasited_times, max_time)
        periodogram_list += [periodogram]
    print("Nb points: ", n / repetitions)
    end_time = time.time()
    print("Periodogram time:", end_time - start_time)

    print("Estimation in full model:")
    print('|' + '-' * (repetitions) + '|')
    print('|', end='')
    start_time = time.time()
    with Pool(5) as p:
        estimations = np.array(p.starmap(job, zip(range(repetitions), periodogram_list, [max_time]*(repetitions))))
    print('|\n Done')
    end_time = time.time()
    np.savetxt("saved_estimations/" + str(repetitions//5) + "partition_2neurons_estimation_NlogNs.csv", estimations, delimiter=",")

    for l, mask in enumerate(mask_list_id):
        print("")
        print("Estimation in reduced model:")
        print('|' + '-' * (repetitions) + '|')
        print('|', end='')
        start_time = time.time()
        with Pool(5) as p:
            reduced_estimations = np.array(p.starmap(reduced_job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions), [mask] * (repetitions))))
        print('|\n Done')
        # estimations = np.array([job(it, periodo) for it, periodo in zip(range(repetitions), periodogram_list)])
        end_time = time.time()
        print("Estimation time:", end_time - start_time)
        np.savetxt("saved_estimations/" + str(repetitions//5) + "partition_2neurons_estimation_NlogN_id" + str(l+1) + ".csv", reduced_estimations, delimiter=",")

    for l, mask in enumerate(mask_list_nonid):
        print("")
        print("Estimation in reduced model:")
        print('|' + '-' * (repetitions) + '|')
        print('|', end='')
        start_time = time.time()
        with Pool(5) as p:
            reduced_estimations = np.array(p.starmap(reduced_job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions), [mask] * (repetitions))))
        print('|\n Done')
        end_time = time.time()
        print("Estimation time:", end_time - start_time)
        np.savetxt("saved_estimations/" + str(repetitions//5) + "partition_2neurons_estimation_NlogN_nonid" + str(l+1) + ".csv", reduced_estimations, delimiter=",")

