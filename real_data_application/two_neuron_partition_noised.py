import pandas as pd
import numpy as np
from class_and_func.spectral_functions import fast_multi_periodogram, ll_unnoised_mask
from class_and_func.estimator_class import multivariate_spectral_noised_estimator, multivariate_spectral_unnoised_estimator
from multiprocessing import Pool
import time


def job(it, periodo, max_time):
    np.random.seed(it)
    estimator = multivariate_spectral_noised_estimator()
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


def job_mask(it, periodo, max_time):
    mask = np.array([[True, False],
                     [True,  True]])
    np.random.seed(it)
    estimator = multivariate_spectral_noised_estimator(mask=mask)
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


def unnoised_job(it, periodo, max_time):
    np.random.seed(it)
    estimator = multivariate_spectral_unnoised_estimator()
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x

if __name__ == "__main__":
    df = pd.read_csv('spk_mouse22.csv', sep=',', header=0, index_col=0)
    #print(df.values)
    which="unnoised"

    max_time = 725
    partition_size = 725//5
    partition_list = [partition_size * i for i in range(0, max_time // partition_size +1)]
    print(partition_list)

    dim = 2
    repetitions = 5
    # K_func = lambda x : int(x)
    K_func = lambda x: int(x * np.log(x))

    data = [df.values[(df.values[:, 2] == i) * (df.values[:, 1] > 4.0), 0:2] for i in range(1,6)]
    variates_idx = {5.0:1, 6.0:2}
    data = [np.array([(t - i * max_time, variates_idx[m]) for (t,m) in data[i]]) for i in range(0,5)]
    print(data[0][:,0])

    partitioned_data = []

    for i in range(len(partition_list) - 1):
        for parasited_times in data:
            flag1 = parasited_times[:, 0] <= partition_list[i + 1]
            flag2 = partition_list[i] < parasited_times[:, 0]

            aux_parasited = parasited_times[flag1 * flag2, :]
            partitioned_data += [aux_parasited]

    print(len(partitioned_data))
    # Data count
    print("# of points per repetition:", [len(u) for u in partitioned_data])

    for u in partitioned_data:
        aux = np.array(u)
        nb_1, nb_2 = np.sum(aux[:,1] == 1), np.sum(aux[:,1] == 2)
        print("# of points per dimension:", nb_1, nb_2)

    # Periodogram
    periodogram_list = [fast_multi_periodogram(K_func(len(u)), u, max_time) for u in partitioned_data]

    print("K:", [K_func(len(u)) for u in periodogram_list])
    partitioned_total = len(partitioned_data)
    #
    # res_red = np.zeros((repetitions, 14))
    print(periodogram_list[-1])
    print("|" + "-"*partitioned_total + "|")
    print("|", end="")
    start_time = time.time()
    if which == "full":
        text = ""
        with Pool(4) as p:
            estimations = np.array(
                p.starmap(job, zip(range(partitioned_total), periodogram_list, [partition_size] * partitioned_total)))
    elif which == "reduced":
        text = "_red"
        with Pool(5) as p:
            estimations = np.array(
                p.starmap(job_mask, zip(range(partitioned_total), periodogram_list, [partition_size] * partitioned_total)))
    elif which == "unnoised":
        text = "_unnoised"
        with Pool(5) as p:
            estimations = np.array(
                p.starmap(unnoised_job, zip(range(partitioned_total), periodogram_list, [partition_size] * partitioned_total)))
        # for i in range(partitioned_total):
        #    print("*"*150)
        #    print(i)
        #    estimations = unnoised_job(i, periodogram_list[i], partition_size)
    else:
        print("wrong")
    #for it, periodo in zip(range(partitioned_total), periodogram_list):
    #    res[it, :] = job_mask(it, periodo, max_time)
    end_time = time.time()
    print("|")
    print("In ", end_time - start_time, " s.")
    # print(res_red)

    np.savetxt("saved_estimations/" + str(max_time//partition_size) + "partition_2neurons_estimation_NlogN" + text + ".csv", estimations, delimiter=",")