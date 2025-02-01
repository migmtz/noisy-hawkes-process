import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import sys

sns.set_theme()
sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 16, 'axes.labelsize': 14,
                             'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })


if __name__ == "__main__":

    parameters_col = [r"$\mu_1$", r"$\mu_2$", r"$\alpha_{11}$", r"$\alpha_{21}$", r"$\beta_1$", r"$\beta_2$",
                      r"$\lambda_0$"]
    parameters_tri = [r"$\mu_1$", r"$\mu_2$", r"$\alpha_{11}$", r"$\alpha_{21}$", r"$\alpha_{22}$", r"$\beta_1$",
                      r"$\beta_2$", r"$\lambda_0$"]
    parameters_full = [r"$\mu_1$", r"$\mu_2$", r"$\alpha_{11}$", r"$\alpha_{12}$", r"$\alpha_{21}$", r"$\alpha_{22}$",
                       r"$\beta_1$", r"$\beta_2$", r"$\lambda_0$"]

    # Column interactions boxplots

    theta_col = np.array([1., 1., 0.5, 0.4, 1., 1.3, 0.5])
    theta_tri = np.array([1., 1., 0.5, 0.4, 0.4, 1., 1.3, 0.5])

    theta_colfull = np.array([1., 1., 0.5, 0.0, 0.4, 0.0, 1., 1.3, 0.5])
    theta_trifull = np.array([1., 1., 0.5, 0.0, 0.4, 0.4, 1., 1.3, 0.5])

    est_col = pd.read_csv('../simulated_data/saved_estimations/multivariate_column_3000.csv', sep=',', header=None, index_col=None).to_numpy()
    est_colmax = pd.read_csv('../simulated_data/saved_estimations/multivariate_column_red_3000.csv', sep=',', header=None, index_col=None).to_numpy()
    est_tri = pd.read_csv('../simulated_data/saved_estimations/multivariate_triangle_3000.csv', sep=',', header=None, index_col=None).to_numpy()
    est_trimax = pd.read_csv('../simulated_data/saved_estimations/multivariate_triangle_red_3000.csv', sep=',', header=None, index_col=None).to_numpy()

    fig, ax = plt.subplots(2, 1, figsize=(5.90666 * 2, 10), sharey=True)
    for axs in ax:
        axs.remove()

    gridspec = ax[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    estimations_list = [[est_colmax, est_col], [est_trimax, est_tri]]
    parameters_list = [[theta_col, parameters_col, theta_colfull], [theta_tri, parameters_tri, theta_trifull]]

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"Scenario {row + 1}")

        ax = subfig.subplots(1, 2, sharey=True)
        bp = ax[0].boxplot(estimations_list[row][0], showmeans=True, meanprops={"markersize": 5})
        rp = ax[0].scatter(range(1, len(parameters_list[row][0]) + 1), parameters_list[row][0], label="True parameter",
                           s=5 ** 2)

        ax[0].set_xticklabels(parameters_list[row][1])
        ax[0].legend([rp, bp['means'][0]], ['True parameter', 'Mean estimation'], loc='best')
        ax[0].set_title(r"Reduced Model ($\mathcal{Q}_\Lambda$)")

        bp = ax[1].boxplot(estimations_list[row][1], showmeans=True, meanprops={"markersize": 5})
        rp = ax[1].scatter(range(1, 10), parameters_list[row][2], label="True parameter", s=5 ** 2)

        aux = np.max(est_col[est_col[:, -2] < np.max(est_col[:, -2]), -2])
        # ax[1].set_ylim([0.0 - 0.05 * aux,
        #             1.05 * aux])
        ax[1].set_xticklabels(parameters_full)
        ax[1].legend([rp, bp['means'][0]], ['True parameter', 'Mean estimation'], loc='best')
        ax[1].set_title(r"Full Model ($\mathcal{Q}$)")

        ax[0].set_ylim((-0.05, 2.05))

    fig.savefig("graphics/column_triangle_model_estimation.pdf", format="pdf", bbox_inches="tight")
    plt.show()
