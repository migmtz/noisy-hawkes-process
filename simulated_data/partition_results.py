import numpy as np
from scipy.linalg import norm
import pandas as pd


if __name__ == "__main__":
    max_time = 8000
    partition_size = 2000
    repetitions = max_time // partition_size
    print(repetitions)

    N_list = ["N", "NlogN"]
    noise = 2.0
    parameters = ["mu", "alpha", "beta", "noise"]

    mu = 1.0
    alpha = 0.5
    beta = 1.0
    theta = np.array([mu, alpha, beta, noise])

    estimations = np.zeros((repetitions, len(N_list), len(parameters), 3))

    for id_N, N in enumerate(N_list):
        for id_param, parameter in enumerate(parameters):
            estimations[:, id_N, id_param, :] = pd.read_csv('saved_estimations_univariate/partition/' + str(partition_size) + 'univariate_partition_' + N + '_' + parameter + '_' + str(np.round(noise, 2)) + '.csv', sep=',', header=None,
                                             index_col=None).to_numpy()[:, :-1]


            print(N, parameter, theta[[i!=id_param for i in range(4)]], np.mean(estimations[:, id_N, id_param, :], axis=0))