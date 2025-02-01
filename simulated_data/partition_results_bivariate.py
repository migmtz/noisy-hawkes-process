import numpy as np
from scipy.linalg import norm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()

sns.set_context("paper", rc={"font.size":14,"axes.titlesize":14, 'axes.labelsize': 14,
                             'xtick.labelsize': 14,'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })


if __name__ == "__main__":
    dim = 2
    max_time = 6000
    partition_size = 300
    repetitions = max_time // partition_size
    print(repetitions)

    parameters_full = [r"$\mu_1$", r"$\mu_2$", r"$\alpha_{11}$", r"$\alpha_{12}$", r"$\alpha_{21}$", r"$\alpha_{22}$",
                       r"$\beta_1$", r"$\beta_2$", r"$\lambda_0$"]

    noise = 0.5

    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.5, 0.0],
                      [0.4, 0.4]])
    beta = np.array([[1.0],
                     [1.3]])

    theta = np.concatenate((mu.ravel(), alpha.ravel(), beta.ravel(), np.array([noise])))

    estimations = pd.read_csv('saved_estimations/partition/' + str(partition_size) + 'multivariate_column_partition_' + str(max_time) + '.csv', sep=',', header=None,
                                     index_col=None).to_numpy()

    print(np.mean(estimations[:, :], axis=0))
    print("Quantiles of interaction Scenario 1", np.quantile(estimations[:, dim:dim + dim**2], axis=0, q=0.05))#, method="closest_observation"))
    print("Proportion of null Scenario 1:", np.mean(estimations < 2e-16, axis=0))

    theta = np.concatenate((mu.ravel(), alpha.ravel(), beta.ravel(), np.array([noise])))


    estimations = pd.read_csv(
        'saved_estimations/partition/' + str(partition_size) + 'multivariate_triangle_partition_' + str(
            max_time) + '.csv', sep=',', header=None,
        index_col=None).to_numpy()

    print(np.mean(estimations[:, :], axis=0))
    print("Quantiles of interaction Scenario 1", np.quantile(estimations[:, dim:dim + dim**2], axis=0, q=0.05))#, method="closest_observation"))
    print("Proportion of null Scenario 2:", np.mean(estimations < 2e-16, axis=0))

    ################################################################## Red
    # estimations_red = np.zeros((repetitions, 7))
    #
    # estimations_red[:, :] = pd.read_csv(
    #     'saved_estimations/partition/' + str(partition_size) + 'multivariate_column_partition_red_' + str(
    #         max_time) + '.csv', sep=',', header=None,
    #     index_col=None).to_numpy()
    #
    # print(np.mean(estimations_red[:, :], axis=0))
    #
    # fig, ax = plt.subplots(1, 2, figsize=(5.90666*2, 6), sharey=True)
    #
    # bplot = ax[0].boxplot(estimations)
    # _ = ax[0].scatter(range(1, 10), theta, marker="*", s=100)
    # ax[0].set_xticklabels(parameters_full)
    #
    # print("Proportion of null:", np.mean(estimations < 2e-16, axis=0))
    # _ = fig.suptitle("Partition method")
    # #ax[0].set_ylim((-0.1, 2.0))
    #
    # alpha_mask = np.array([[True, False],
    #                        [True, False]])
    #
    # mask = np.array([True] * (dim * (2 + dim) + 1))
    # mask[dim:dim + dim * dim] = alpha_mask.ravel()
    #
    # positions_red = np.arange(9)[mask]
    #
    # bplot = ax[1].boxplot(estimations_red)
    # print(theta[mask])
    # _ = ax[1].scatter(range(1, 8), theta[mask], marker="*", s=100, label="True parameter")
    # ax[1].set_xticklabels(np.array(parameters_full)[mask])
    #
    # single_estimation = pd.read_csv(
    #     'saved_estimations/partition/4000multivariate_column_partition_red_' + str(
    #         max_time) + '.csv', sep=',', header=None,
    #     index_col=None).to_numpy()
    #
    # print(single_estimation)
    #
    # _ = ax[1].scatter(range(1, 8), single_estimation, marker="o", s=50, label="Single estimation")
    #
    # ax[1].legend()

    # ax[0].set_title("Full model $\mathcal{Q}$")
    # ax[1].set_title("Reduced model $\mathcal{Q}_{\Lambda}$")
    # plt.savefig("partition_estimation_boxplots.pdf", format="pdf", bbox_inches="tight")

    plt.show()