import numpy as np
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram
from multiprocessing import Pool
import time



def job(it, periodo, max_time):
    np.random.seed(it)
    mask = np.array([[True, True],
                     [True, True]])
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


def f_1(n):
    # M = N(T)
    return int(n)


def f_2(n):
    # M = N(T) log N(T)
    return int(n * np.log(n))


sns.set_context("paper", rc={"font.size":14,"axes.titlesize":16, 'axes.labelsize': 14,
                             'xtick.labelsize': 14,'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })


if __name__ == "__main__":

    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.5, 0.0],
                      [0.4, 0.0]])
    beta = np.array([[1.0],
                     [1.3]])

    #noise_list = mu[0]/(1 - alpha[0]) * np.array([0.2 * k for k in range(1, 11)])
    noise_list = [0.5]
    print(noise_list)

    burn_in = -100

    max_time = 6000
    partition_size = 300
    partition_list = [partition_size * i for i in range(0, max_time // partition_size +1)]
    print(partition_list)

    K_func_list = [f_1]
    K_func_name = ["N"]

    for noise in noise_list:

        # Simulations and periodograms

        np.random.seed(0)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps
        print("lenH", len(hp_times), max_time * mu / (1 - alpha))

        pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time,
                                             burn_in=burn_in)
        pp.simulate()
        pp_times = pp.timestamps
        print("lenP", len(pp_times), max_time * noise)

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]

        print(mu,alpha,beta,noise)


        for K_idx, K_func in enumerate(K_func_list):
            periodogram_list = []
            for i in range(len(partition_list) - 1):
                flag1 = parasited_times[:, 0] <= partition_list[i + 1]
                flag2 = partition_list[i] < parasited_times[:, 0]

                aux_parasited = parasited_times[flag1 * flag2, :]
                K = K_func(aux_parasited.shape[0])
                periodogram = fast_multi_periodogram(K, aux_parasited, partition_size)

                periodogram_list += [periodogram]

            repetitions = len(partition_list) - 1
            print('|' + '-' * (repetitions) + '|')
            print('|', end='')
            start_time = time.time()
            with Pool(4) as p:
                estimations = np.array(
                    p.starmap(job, zip(range(repetitions), periodogram_list, [partition_size] * (repetitions))))
            print('|\n Done')
            end_time = time.time()
            print("Estimation time:", end_time - start_time)
            # print(estimations)
            print("*" * 100)

            np.savetxt("saved_estimations/partition/" + str(partition_size) + "multivariate_column_partition_" + str(max_time) + ".csv", estimations,
                       delimiter=",")
