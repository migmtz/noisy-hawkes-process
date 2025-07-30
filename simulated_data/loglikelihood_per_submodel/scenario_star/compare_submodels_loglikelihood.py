import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from class_and_func.spectral_functions import fast_multi_periodogram, grad_ll_mask
from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_spectral_noised_estimator, \
    multivariate_spectral_unnoised_estimator

sns.set_theme()

sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 16, 'axes.labelsize': 14,
                             'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })

mask_list = [np.array([[True, True],
                       [True, True]]),
             np.array([[True, False],
                       [True, False]]),
             np.array([[False, True],
                       [False, True]]),
             np.array([[True, False],
                       [True, True]]),
             np.array([[True, True],
                       [False, True]]),
             np.array([[True, False],
                       [False, True]]),
             np.array([[True, True],
                       [False, False]]),
             np.array([[False, False],
                       [True, True]])
             ]

if __name__ == "__main__":

    dim = 2

    nb_partition = 5

    ###### BEFORE
    mu = np.array([[0.01],
                   [0.15]])
    alpha = np.array([[0.80, 0.00],
                      [3.5, 0.70]])
    beta = np.array([[15.0],
                     [35.0]])
    noise = 0.05

    real_theta = [mu, alpha, beta, np.array([noise])]

    max_time = 725 / nb_partition
    burn_in = -100
    repetitions = 5 * nb_partition

    K_func = lambda x: int(x * np.log(x))

    ####

    # Simulations and periodograms

    periodogram_list = []
    K_list = []

    n = 0

    for it in range(repetitions):
        np.random.seed(it+50)
        hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
        hp.simulate()
        hp_times = hp.timestamps

        pp = multivariate_exponential_hawkes(noise * np.ones((2, 1)), 0 * alpha, beta, max_time=max_time,
                                             burn_in=burn_in)
        pp.simulate()
        pp_times = pp.timestamps

        idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
        parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
        n += len(parasited_times)

        K = K_func(parasited_times.shape[0])
        K_list += [K]
        periodogram = fast_multi_periodogram(K, parasited_times, max_time)
        periodogram_list += [periodogram]


    ###### Loglikelihoods

    print("(Negative) Log-likelihoods: \n")

    noisedNlogN = pd.read_csv('saved_estimations/' + str(nb_partition) + 'partition_2neurons_estimation_NlogN.csv', sep=',', header=None, index_col=None).to_numpy()
    boxplots = noisedNlogN[:repetitions * nb_partition, -1]
    estimations = np.mean(noisedNlogN[:repetitions * nb_partition, :-1], axis=0)
    ll = np.mean([grad_ll_mask(estimations, periodogram, K, max_time, mask=None)[0] for periodogram, K in
                  zip(periodogram_list, K_list)])
    print("Complete: ", ll)  # , "\n", noisedNlogN[:, 2:6])

    for i in range(4):
        noisedNlogN = pd.read_csv('saved_estimations/' + str(nb_partition) + 'partition_2neurons_estimation_NlogN_id' + str(i + 1) + '.csv', sep=',',
                                  header=None, index_col=None).to_numpy()
        boxplots = np.vstack((boxplots, noisedNlogN[:repetitions * nb_partition, -1]))
        estimations = np.mean(noisedNlogN[:repetitions * nb_partition, :-1], axis=0)

        ll = np.mean(
            [grad_ll_mask(estimations, periodogram, K, max_time, mask=mask_list[i + 1])[0] for periodogram, K in
             zip(periodogram_list, K_list)])
        print("Id" + str(i + 1) + ": ", ll)  # , "\n", noisedNlogN[:repetitions, 2:2+np.sum(mask_list[i+1])])

    for i in range(3):
        noisedNlogN = pd.read_csv('saved_estimations/' + str(nb_partition) + 'partition_2neurons_estimation_NlogN_nonid' + str(i + 1) + '.csv', sep=',',
                                  header=None, index_col=None).to_numpy()
        boxplots = np.vstack((boxplots, noisedNlogN[:repetitions * nb_partition, -1]))
        estimations = np.mean(noisedNlogN[:repetitions * nb_partition, :-1], axis=0)
        ll = np.mean(
            [grad_ll_mask(estimations, periodogram, K, max_time, mask=mask_list[i + 5])[0] for periodogram, K in
             zip(periodogram_list, K_list)])
        print("Nonid" + str(i + 1) + ": ", ll)  # , "\n", noisedNlogN[:repetitions, 2:2+np.sum(mask_list[i+5])])
