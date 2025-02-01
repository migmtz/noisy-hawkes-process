import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import norm


sns.set_theme()

def string_csv(version, horizon, N, parameter, noise):
    res = ('saved_estimations_univariate/horizons_' + version + '/'
           + str(horizon) + 'univariate_horizon_' + N + '_' + parameter + '_' + str(noise) + '.csv')
    return res


def plot_with_confidence(x, y_down, y_up, ax, c='blue', alpha=0.2):
    ax.plot(x, y_down, '--', color=c, alpha=alpha*1.5)
    ax.plot(x, y_up, '--', color=c, alpha=alpha*1.5)
    ax.fill_between(x, y_down, y_up, color=c, alpha=alpha)


linestyles = ["solid", "dashdot", "dotted"]

sns.set_context("paper", rc={"font.size":14,"axes.titlesize":16, 'axes.labelsize': 14,
                             'xtick.labelsize': 14,'ytick.labelsize': 14, 'legend.fontsize': 10,
                             })


linestyles = ["solid", "dashdot", "dotted"]
titles = [r"$\mu$ fixed", r"$\alpha$ fixed", r"$\beta$ fixed", r"$\lambda_0$ fixed"]
labels = [r"$M=N(T)$",
          r"$M=N(T)\,\log N(T)$", "Adaptive"]


if __name__ == "__main__":
    parameter_list = ["mu", "alpha", "beta", "noise"]
    horizons = [250, 500, 1000, 2000, 4000, 8000]
    N_func = ["N", "NlogN", "2adaptive"]
    versions = ["revision"]#["original", "revision"]
    repetitions = 50

    mu_ = {"original":1.0, "revision":1/8}
    alpha_ = {"original":0.5, "revision":1/8}
    beta_ = {"original":1.0, "revision":1.0}
    noise_length = 10
    noise_chosen = 4

    for id, version in enumerate(versions):
        mu = mu_[version]
        alpha = alpha_[version]
        beta = beta_[version]
        noise_list = np.round(mu * np.array([0.2 * k for k in range(1, noise_length+1)]) / (1-alpha), 2)
        print(noise_list[noise_chosen])
        estimations = np.zeros((repetitions, len(parameter_list),len(horizons), len(N_func), len(noise_list), 3))
        computation_times = np.zeros((repetitions, len(parameter_list),len(horizons), len(N_func), len(noise_list)))
        errors = np.zeros((repetitions, len(parameter_list),len(horizons), len(N_func), len(noise_list)))
        mean = np.zeros((len(parameter_list),len(horizons), len(N_func), len(noise_list)))
        std_dev = np.zeros((len(parameter_list),len(horizons), len(N_func), len(noise_list)))

        for id_noise, noise in enumerate(noise_list):
            theta = np.array([mu, alpha, beta, noise])
            for id_param, parameter in enumerate(parameter_list):
                theta_fixed = theta[[i != id_param for i in range(4)]]
                for id_horizon, horizon in enumerate(horizons):
                    for id_N, N in enumerate(N_func):
                            aux = pd.read_csv(string_csv(version, horizon, N, parameter, noise),
                                              sep=',', header=None, index_col=None).to_numpy()

                            estimations[:, id_param, id_horizon, id_N, id_noise, :] = aux[:, :-1]
                            #print(theta_fixed, np.mean(aux[:, :-1], axis=0))
                            computation_times[:, id_param, id_horizon, id_N, id_noise] = aux[:, -1]
                            print("*"*100)
                            print(aux[:,:-1], theta_fixed, norm(aux[:,:-1] - theta_fixed,axis=1) )
                            print("*" * 100)
                            aux_error = norm(aux[:, :-1] - theta_fixed,axis=1) / norm(theta_fixed)
                            print(aux_error)
                            errors[:, id_param, id_horizon, id_N, id_noise] = aux_error
                            mean[id_param, id_horizon, id_N, id_noise] = np.mean(aux_error)
                            std_dev[id_param, id_horizon, id_N, id_noise] = np.std(aux_error)

    print("Noise: ", noise_list[6])
    fig, ax = plt.subplots(1, 4, figsize=(5.90666 * 2, 4), sharey=True)
    for id_param, parameter in enumerate(parameter_list):
        for id_N, N in enumerate(N_func):
            ax[id_param].plot(horizons, mean[id_param, :, id_N, noise_chosen], c="C" + str(id_N),
                                 linestyle=linestyles[0], label=labels[id_N])
            aux1 = mean[id_param, :, id_N, noise_chosen] - 1.96 * np.sqrt(std_dev[id_param, :, id_N, noise_chosen] / repetitions)
            aux2 = mean[id_param, :, id_N, noise_chosen] + 1.96 * np.sqrt(std_dev[id_param, :, id_N, noise_chosen] / repetitions)
            plot_with_confidence(horizons, aux1, aux2, ax[id_param], c="C" + str(id_N), alpha=0.1)
            ax[id_param].set_xscale("log")
            #ax[1, idx_parameter].set_xscale("log")
            ax[id_param].set_xlabel(r"$T$")
            ax[id_param].set_ylabel(r"$\ell^2$ relative error")
            ax[id_param].set_title(titles[id_param])
            ax[id_param].legend()


    ax[0].set_ylim((ax[0].get_ylim()[0], 3.0))
    ax[0].legend()


    # fig.suptitle(r"Noise level: $\lambda_0 = {}$".format(noise_levels[idx]))
    plt.tight_layout()

    plt.savefig("l2_error_wrt_nbpoints_adaptive.pdf", format="pdf", bbox_inches="tight")


    plt.show()