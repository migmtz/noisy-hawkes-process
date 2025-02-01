import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_theme()

sns.set_context("paper", rc={"font.size":14,"axes.titlesize":16, 'axes.labelsize': 14,
                             'xtick.labelsize': 14,'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })


if __name__ == "__main__":

    dim = 3
    positions = [0, 1, 2] + [4 + i for i in range(dim ** 2)] + [5 + i for i in range(dim ** 2, dim ** 2 + dim)]
    positions += [positions[-1] + 2]
    positions = np.array(positions)
    print(positions)

    indices = [0, dim, dim + dim*dim, len(positions)-1, len(positions)]
    print(indices)

    alpha_mask = np.array([[False, True,  True ],
                           [True,  True,  True ],
                           [False, False, True ]])

    mask = np.array([True]*(dim * (2 + dim) + 1))
    mask[dim:dim + dim*dim] = alpha_mask.ravel()

    positions_red = positions[mask]
    print(positions_red)
    #positions_red = np.array(positions_red)
    indices_red = [0, dim, dim + int(np.sum(alpha_mask)), 2 * dim + int(np.sum(alpha_mask)),2 * dim + int(np.sum(alpha_mask)) + 1]

    labels = [["$\\mu_1$", "$\\mu_2$", "$\\mu_3$"]]

    labels1 = ["$\\alpha_{11}$", "$\\alpha_{12}$", "$\\alpha_{13}$"]
    labels1 += ["$\\alpha_{21}$", "$\\alpha_{22}$", "$\\alpha_{23}$"]
    labels1 += ["$\\alpha_{31}$", "$\\alpha_{32}$", "$\\alpha_{33}$"]
    labels += [labels1]

    labels += [["$\\beta_1$", "$\\beta_2$", "$\\beta_3$"]]

    labels += [["$\\lambda_0$"]]

    fig, ax = plt.subplots(1, 4, figsize=(20, 8),width_ratios=(3,9,3,1))

    #noisedN = pd.read_csv('saved_estimations/realdata_estimation_noised_N.csv', sep=',', header=None, index_col=None)
    #bplot = ax.boxplot(noisedN, positions=positions, widths=0.2, tick_labels=[""]*16,
    #                  patch_artist=True, label="N")
    #for patch in bplot['boxes']:
    #   patch.set_facecolor("paleturquoise")

    noisedNlogN = pd.read_csv('saved_estimations/realdata_estimation_noised_NlogN.csv', sep=',', header=None, index_col=None).to_numpy()
    for i in range(4):
        #print(i)
        bplot = ax[i].boxplot(noisedNlogN[:, indices[i]:indices[i+1]], widths=0.2, positions=positions[indices[i]:indices[i+1]], tick_labels=labels[i],
                           patch_artist=True, label=r"$\mathcal{Q}$")
        for patch in bplot['boxes']:
            patch.set_facecolor("paleturquoise")
    alpha_est_noised = noisedNlogN[:, dim:dim+dim**2].reshape((5,3,3))
    mean = np.mean(noisedNlogN, axis=0)
    print(np.round(mean,2), np.round(np.std(noisedNlogN, axis=0, ddof=1), 2))
    #print([np.max(np.abs(np.linalg.eig(u)[0])) for u in alpha_est_noised])
    #print("noisedNlogN", np.quantile(alpha_est_noised, axis=0, q=0.05, method="closest_observation"))
    #print("noisedNlogN", alpha_est_noised)
    print("noisedNlogN percentages: ", np.mean(alpha_est_noised < 2e-16, axis=0))

    noisedNlogN = pd.read_csv('saved_estimations/realdata_estimation_noised_red_NlogN_anna.csv', sep=',', header=None,
                              index_col=None).to_numpy()
    for i in range(4):
        dat = noisedNlogN[:, indices_red[i]:indices_red[i+1]]
        bplot = ax[i].boxplot(dat, widths=0.2, positions=positions_red[indices_red[i]:indices_red[i+1]] + 0.2, tick_labels=[""]*(dat.shape[1]),
                           patch_artist=True, label=r"$\mathcal{Q}_{\Lambda}$")
        for patch in bplot['boxes']:
            patch.set_facecolor("cornflowerblue")

    alpha_est_noised = noisedNlogN[:, dim:dim + dim ** 2].reshape((5, 3, 3))
    mean = np.mean(noisedNlogN, axis=0)
    print(np.round(mean,2), np.round(np.std(noisedNlogN, axis=0, ddof=1), 2))
    #print([np.max(np.abs(np.linalg.eig(u)[0])) for u in alpha_est_noised])
    print("noisedNlogN reduced percentages: ", np.mean(alpha_est_noised < 2e-16, axis=0))

    unnoisedNlogN = pd.read_csv("saved_estimations/realdata_estimation_unnoised_NlogN.csv", sep=',', header=None, index_col=None).to_numpy()
    for i in range(3):
        dat = unnoisedNlogN[:, indices[i]:indices[i + 1]]
        bplot = ax[i].boxplot(dat, widths=0.2,
                              positions=positions[indices[i]:indices[i + 1]] + 0.5, tick_labels=[""] * (dat.shape[1]),
                              patch_artist=True, label="Unnoised Hawkes model")
        for patch in bplot['boxes']:
            patch.set_facecolor("slategrey")
    alpha_est_unnoised = unnoisedNlogN[:, dim:dim + dim ** 2].reshape((5, 3, 3))
    mean = np.mean(unnoisedNlogN, axis=0)
    print(np.round(mean,2), np.round(np.std(unnoisedNlogN, axis=0, ddof=1), 2))

    ax[1].legend()
    ax[1].set_ylim((-0.1, 2.5))

    plt.savefig("real_data_estimation_boxplots.pdf", format="pdf", bbox_inches="tight")

    plt.show()

