import numpy as np
from multiprocessing import Pool, cpu_count
import time
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram
import pickle


def job(it, periodo, max_time):
    np.random.seed(it)
    mask = np.array([[True, True],
                     [True, True]])
    res = []
    for periodogram in periodo:
        estimator = multivariate_spectral_noised_estimator(mask=mask)
        aux = estimator.fit(periodogram, max_time)
        res += [aux.x]

    print('-', end='')

    return res


def spike_and_slab(dim, seed=None):
    if seed is None:
        seed0 = np.random.randint(0,100000)
    else:
        seed0 = seed

    np.random.seed(seed0)
    p = 0.66
    alpha = np.random.chisquare(2, (dim, dim))
    mask = np.random.binomial(1, p, (dim, dim))

    alpha = mask * alpha
    radius = np.max(np.abs(np.linalg.eig(alpha)[0]))
    if radius == 0.0:
        fact = 1.0
    else:
        div = np.random.uniform(1e-16, 1 - 1e-16)
        fact = div / radius
    init_alpha = alpha * fact

    init_alpha[init_alpha <= 1e-1] = 0.0

    return init_alpha


if __name__ == "__main__":

    repetitions = 100 #* 10 #15*10 done, change range if necessary for new estimations and to make 160.
    max_time = 200
    burn_in = -100
    partition_nb = 10
    points_nb = 15000

    K_func = lambda x: int(x)
    periodogram_list = []
    theta_list = []
    max_time_list = []

    for it in range(repetitions):
        np.random.seed(it *100)
        flag = False
        while not(flag):
            mu = np.random.chisquare(2, (2, 1))
            flag = 0.5 < mu[0] / mu[1] < 2.0
        alpha = spike_and_slab(2, it)
        flag = False
        while not (flag):
            beta = np.random.chisquare(2, (2,1)) + 0.5
            flag = 0.5 < beta[0] / beta[1] < 2.0
        noise = np.random.chisquare(2) * np.ones((2,1)) + 0.1

        theta = np.concatenate((mu.ravel(), alpha.ravel(), beta.ravel(), noise[0]))
        print(theta)

        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time,
                                             burn_in=burn_in)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

        full_horizon = max_time * (points_nb*1.5)/len(parasited_times)

        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=full_horizon, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=full_horizon,
                                             burn_in=burn_in)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
        if parasited_times.shape[0] < points_nb+1:
            raise ValueError('Not enough.')
        complete_parasited = parasited_times[0:points_nb+1]

        new_max_time = complete_parasited[-1][0]
        partition_size = new_max_time / partition_nb
        partition_list = [i * partition_size for i in range(partition_nb + 1)]
        #print(new_max_time, partition_list)

        periodogram_inter = []
        for i in range(len(partition_list) - 1):
            flag1 = complete_parasited[:, 0] <= partition_list[i + 1]
            flag2 = partition_list[i] <= complete_parasited[:, 0]

            aux_parasited = complete_parasited[flag1 * flag2, :]
            K = K_func(aux_parasited.shape[0])
            periodogram = fast_multi_periodogram(K, aux_parasited, partition_size)

            periodogram_inter += [periodogram]

        theta_list += [theta]
        max_time_list += [partition_size]
        periodogram_list += [periodogram_inter]
    #toSave = np.concatenate((theta_list, np.array([K_list]).T), axis=1)
    #np.savetxt("saved_estimations/spike_and_slab/parameters_K.csv", toSave, delimiter=",")
    with open("saved_estimations/spike_and_slab/prueba_parameters_fixed", 'wb') as file:
        pickle.dump(np.array(theta_list), file)

    print('|' + '-' * (repetitions) + '|')
    print('|', end='')
    start_time = time.time()
    with Pool(8) as p:
        estimations = np.array(p.starmap(job, zip(range(repetitions), periodogram_list, max_time_list)))
    print('|\n Done')
    # estimations = np.array([job(it, periodo) for it, periodo in zip(range(repetitions), periodogram_list)])
    end_time = time.time()
    print("Estimation time:", end_time - start_time)

    print(estimations, estimations.shape)

    #np.savetxt("saved_estimations/spike_and_slab/prueba.csv", estimations, delimiter=",")

    with open("saved_estimations/spike_and_slab/prueba_estimations_fixed", 'wb') as file:
        pickle.dump(estimations, file)
    #
    # with open("saved_estimations/spike_and_slab/prueba.csv", 'rb') as file:
    #     blah = pickle.load(file)
    #
    # print(blah)