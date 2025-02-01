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
    #print('-', end='')

    return aux



def f_1(n):
    # M = N(T)
    return int(n)


def f_2(n):
    # M = N(T) log N(T)
    return int(n * np.log(n))


if __name__ == "__main__":

    mu = np.array([[1/8]])
    alpha = np.array([[1/8]])
    beta = np.array([[1.0]])

    noise_list = mu[0]/(1 - alpha[0]) * np.array([0.2 * k for k in range(1, 11)])
    print(noise_list)

    burn_in = -100
    repetitions = 50

    horizons = [250, 500, 1000, 2000, 4000, 8000]
    max_time = horizons[-1]

    K_func_list = [f_1, f_2]
    K_func_name = ["N", "NlogN"]

    fixed_list = ["mu", "alpha", "beta", "noise"]

    for noise in noise_list:

        theta = np.concatenate((mu, alpha, beta, np.array([[noise]])))
        fixed_parameter_list = [(i, theta[i]) for i in range(4)]

        # Simulations and periodograms

        print(mu,alpha,beta,noise)
        simulated_points = []
        start_time = time.time()
        for it in range(repetitions):
            np.random.seed(it)
            hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=max_time, burn_in=burn_in)
            hp.simulate()
            hp_times = hp.timestamps
            #print("lenH", len(hp_times), max_time * mu/(1-alpha))

            pp = multivariate_exponential_hawkes(noise * np.ones((1, 1)), 0 * alpha, beta, max_time=max_time,
                                                 burn_in=burn_in)
            pp.simulate()
            pp_times = pp.timestamps
            #print("lenP", len(pp_times), max_time*noise)

            idx = np.argsort(pp_times[1:-1] + hp_times, axis=0)[:, 0]
            parasited_times = np.array(pp_times[1:-1] + hp_times)[idx]
            simulated_points += [parasited_times]

        for K_idx, K_func in enumerate(K_func_list):
            for horizon in horizons:
                periodogram_list = []
                start_time = time.time()
                for it in range(repetitions):
                    aux_parasited = simulated_points[it][simulated_points[it][:, 0] <= horizon, :]
                    K = K_func(aux_parasited.shape[0])
                    periodogram = fast_multi_periodogram(K, aux_parasited, horizon)

                    periodogram_list += [periodogram]

                end_time = time.time()
                #print("Periodogram time:", end_time - start_time)

                for j, fixed_parameter in enumerate(fixed_parameter_list):
                    #print('|' + '-' * (repetitions) + '|')
                    #print('|', end='')
                    start_time = time.time()
                    with Pool(5) as p:
                        estimations = np.array(
                            p.starmap(old_job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions),[fixed_parameter] * (repetitions))))
                    #print('|\n Done')
                    end_time = time.time()
                    #print("Estimation time:", end_time - start_time)
                    #print(estimations)
                    #print("*"*100)

                    np.savetxt("saved_estimations_univariate/horizons_revision/" + str(horizon) + "univariate_horizon_2" + str(K_func_name[K_idx]) + "_"+ fixed_list[j] + "_" + str(np.round(noise, 2)) + ".csv", estimations,
                               delimiter=",")