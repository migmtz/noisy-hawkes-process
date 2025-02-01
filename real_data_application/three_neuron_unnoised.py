import pandas as pd
import numpy as np
from class_and_func.spectral_functions import fast_multi_periodogram
from class_and_func.estimator_class import multivariate_spectral_unnoised_estimator
import time


def job(it, periodo, max_time):
    np.random.seed(it)
    estimator = multivariate_spectral_unnoised_estimator()
    res = estimator.fit(periodo, max_time)

    print('-', end='')

    return res.x


if __name__ == "__main__":
    df = pd.read_csv('spk_mouse22.csv', sep=',', header=0, index_col=0)
    print(df.values)

    max_time = 725
    dim = 3
    repetitions = 5
    K_func = lambda x : int(x)
    # K_func = lambda x: int(x * np.log(x))

    data = [df.values[df.values[:, 2] == i, 0:2] for i in range(1,6)]
    variates_idx = {4.0:1, 5.0:2, 6.0:3}
    data = [[(t - i * max_time, variates_idx[m]) for (t,m) in data[i]] for i in range(0,5)]

    print("# of points per repetition:", [len(u) for u in data])

    for u in data:
        aux = np.array(u)
        nb_1, nb_2, nb_3 = np.sum(aux[:,1] == 1), np.sum(aux[:,1] == 2), np.sum(aux[:,1] == 3)
        print("# of points per dimension:", nb_1, nb_2, nb_3)

    periodogram_list = [fast_multi_periodogram(K_func(len(u)), u, max_time) for u in data]

    print("K:", [K_func(len(u)) for u in periodogram_list])

    res = np.zeros((repetitions, 2*dim + dim*dim))
    print("|" + "-"*repetitions + "|")
    print("|", end="")
    start_time = time.time()
    for it, periodo in zip(range(repetitions), periodogram_list):
        res[it, :] = job(it, periodo, max_time)
    end_time = time.time()
    print("|")
    print("In ", end_time - start_time, " s.")
    print(res)

    np.savetxt("saved_estimations/realdata_estimation_unnoised_N.csv", res, delimiter=",")