import numpy as np
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import old_univariate_spectral_noised_estimator
from class_and_func.spectral_functions import fast_multi_periodogram
from multiprocessing import Pool
import time


def old_job(it, periodo, max_time, fixed_parameter):
    np.random.seed(it+100)

    estimator = old_univariate_spectral_noised_estimator(fixed_parameter)
    start_time = time.time()
    res = estimator.fit(periodo, max_time)
    end_time = time.time()

    aux = np.concatenate((res.x, np.array([end_time-start_time])))
    print('-', end='')

    return aux


def f_1(n):
    # M = N(T)
    return int(n)


def f_2(n):
    # M = N(T) log N(T)
    return int(n * np.log(n))


if __name__ == "__main__":

    mu = np.array([[1.0]])
    alpha = np.array([[0.5]])
    beta = np.array([[1.0]])

    #noise_list = mu[0]/(1 - alpha[0]) * np.array([0.2 * k for k in range(1, 11)])
    noise_list = [2.0]
    print(noise_list)

    burn_in = -100

    max_time = 8000
    partition_size = 2000
    partition_list = [partition_size * i for i in range(0, max_time // partition_size +1)]
    print(partition_list)

    K_func_list = [f_1, f_2]
    K_func_name = ["N", "NlogN"]

    fixed_list = ["mu", "alpha", "beta", "noise"]

    for noise in noise_list:

        theta = np.concatenate((mu, alpha, beta, np.array([[noise]])))
        fixed_parameter_list = [(i, theta[i]) for i in range(4)]

        # Simulations and periodograms

        np.random.seed(0)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps
        print("lenH", len(hp_times), max_time * mu / (1 - alpha))

        pp = multivariate_exponential_hawkes(noise * np.ones((1, 1)), 0 * alpha, beta, max_time=max_time,
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

            for j, fixed_parameter in enumerate(fixed_parameter_list):
                repetitions = len(partition_list) - 1
                print('|' + '-' * (repetitions) + '|')
                print('|', end='')
                start_time = time.time()
                with Pool(2) as p:
                    estimations = np.array(
                        p.starmap(old_job, zip(range(repetitions), periodogram_list, [partition_size] * (repetitions),
                                           [fixed_parameter] * (repetitions))))
                print('|\n Done')
                end_time = time.time()
                print("Estimation time:", end_time - start_time)
                # print(estimations)
                print("*" * 100)

                np.savetxt("saved_estimations_univariate/partition/" + str(partition_size) + "univariate_partition_" + str(
                    K_func_name[K_idx]) + "_" + fixed_list[j] + "_" + str(np.round(noise, 2)) + ".csv", estimations,
                           delimiter=",")
