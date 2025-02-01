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


if __name__ == "__main__":

    alpha = np.array([[0.5]])
    beta = np.array([[1.0]])

    noise_list = [1.0]#[0.2 * i for i in range(1, 11)]
    avg_total_intensity = 4.0

    max_time = 8000
    burn_in = -100
    repetitions = 50

    K_func = lambda x: int(x)

    fixed_list = ["mu", "alpha", "beta", "noise"]

    for noise in noise_list:
        mu = (1-alpha) * (avg_total_intensity - noise)

        theta = np.concatenate((mu, alpha, beta, np.array([[noise]])))
        fixed_parameter_list = [(i, theta[i]) for i in range(4)]

        # Simulations and periodograms

        periodogram_list = []
        print(mu,alpha,beta,noise)
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

            K = K_func(parasited_times.shape[0])
            periodogram = fast_multi_periodogram(K, parasited_times, max_time)

            periodogram_list += [periodogram]

        end_time = time.time()
        print("Periodogram time:", end_time - start_time)
        print("K =", periodogram_list[0].shape)

        for j, fixed_parameter in enumerate(fixed_parameter_list):
            print('|' + '-' * (repetitions) + '|')
            print('|', end='')
            start_time = time.time()
            with Pool(6) as p:
                estimations = np.array(
                    p.starmap(old_job, zip(range(repetitions), periodogram_list, [max_time] * (repetitions),[fixed_parameter] * (repetitions))))
            print('|\n Done')
            end_time = time.time()
            print("Estimation time:", end_time - start_time)
            print(estimations)
            print("*"*100)

            np.savetxt("saved_estimations_univariate/cst_intensity/8000uni_cst_fixed_"+ fixed_list[j] + "_" + str(np.round(noise, 2)) + ".csv", estimations,
                       delimiter=",")