import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import norm

sns.set_theme()


def plot_with_confidence(x, y_down, y_up, ax, c='blue', alpha=0.2):
    ax.plot(x, y_down, '--', color=c, alpha=alpha*1.5)
    ax.plot(x, y_up, '--', color=c, alpha=alpha*1.5)
    ax.fill_between(x, y_down, y_up, color=c, alpha=alpha)


sns.set_context("paper", rc={"font.size":14,"axes.titlesize":14, 'axes.labelsize': 14,
                             'xtick.labelsize': 14,'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })

if __name__ == "__main__":
    alpha = np.array([[0.5]])
    beta = np.array([[1.0]])

    noise_list = [0.2 * i for i in range(1, 11)]
    avg_total_intensity = 4.0

    max_time = 8000
    burn_in = -100
    repetitions = 50

    K_func = lambda x: int(x)

    fixed_list = ["mu", "alpha", "beta", "noise"]
    label_list = ["$\\mu$ fixed", "$\\alpha$ fixed", "$\\beta$ fixed", "$\\lambda_0$ noise"]

    estimations = np.zeros((repetitions, len(noise_list), len(fixed_list), 3))
    errors = np.zeros((repetitions, len(noise_list), len(fixed_list)))
    means = np.zeros((len(noise_list), len(fixed_list)))
    st_dev = np.zeros((len(noise_list), len(fixed_list)))
    thetas = np.zeros((len(noise_list), len(fixed_list), 3))

    for id_noise, noise in enumerate(noise_list):
        mu = (1-alpha) * (avg_total_intensity - noise)
        print(mu)

        theta = np.concatenate((mu, alpha, beta, np.array([[noise]])))

        for i, fixed in enumerate(fixed_list):

            thetas[id_noise, i, :] = theta[[k!=i for k in range(4)]].ravel()

            column_estimations = pd.read_csv('saved_estimations_univariate/cst_intensity/8000uni_cst_fixed_' + fixed_list[i] + '_' + str(np.round(noise, 2)) + '.csv', sep=',', header=None,
                                             index_col=None).to_numpy()

            estimations[:, id_noise, i, :] = column_estimations[:,:-1]
            errors[:, id_noise, i] = norm(estimations[:, id_noise, i, :] - thetas[id_noise, i, :], axis=-1) / norm(thetas[id_noise, i, :])
            means[id_noise, i] = np.mean(errors[:, id_noise, i], axis=0)
            st_dev[id_noise, i] = (50/49) * (np.mean(errors[:, id_noise, i]**2) - means[id_noise, i]**2)
            print(estimations[:, id_noise,i,:].mean(axis=0))
            #print(errors[0, id_noise, i], estimations[0, id_noise, i, :], thetas[id_noise, i, :])

    #print(estimations.mean(axis=0))

    #avg_estimations = estimations.mean(axis=0)
    #print(avg_estimations)
    #print(avg_estimations[:, 0, :].shape, thetas[:, 0, :].shape)

    fig, ax = plt.subplots(figsize=(5.90666 * 2, 6))

    for i in range(4):
        mean = means[:, i]
        st_d = st_dev[:, i]
        #print(st_d)
        ax.plot(noise_list, mean, label=label_list[i])
        plot_with_confidence(noise_list, mean - 1.96 * np.sqrt(st_d/50), mean + 1.96 * np.sqrt(st_d/50), ax, c="C" + str(i), alpha=0.1)

    ax.set_xticks(noise_list)
    ax.set_xlabel("$\\lambda_0$")
    ax.set_ylabel("$\ell_2$ relative error")
    #ax.set_title("Estitmations at constant average intensity")
    #ax.set_ylim((-0.01,0.5))
    plt.legend()
    plt.savefig("constant_intensity_estimation.pdf", format="pdf", bbox_inches="tight")

    plt.show()