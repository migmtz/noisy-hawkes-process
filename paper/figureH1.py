import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from scipy.linalg import norm


sns.set_theme()

sns.set_context("paper", rc={"font.size":14,"axes.titlesize":16, 'axes.labelsize': 14,
                             'xtick.labelsize': 14,'ytick.labelsize': 14, 'legend.fontsize': 14,
                             })


if __name__ == "__main__":

    dim = 2
    repetitions = 100

    labels = [["$\\mu_1$", "$\\mu_2$"]]

    labels1 = ["$\\alpha_{11}$", "$\\alpha_{12}$"]
    labels1 += ["$\\alpha_{21}$", "$\\alpha_{22}$"]
    labels += [labels1]

    labels += [["$\\beta_1$", "$\\beta_2$"]]

    labels += [["$\\lambda_0$"]]

    with open("../simulated_data/saved_estimations/spike_and_slab/prueba_parameters", 'rb') as file:
        parameters = pickle.load(file)

    with open("../simulated_data/saved_estimations/spike_and_slab/prueba_estimations", 'rb') as file:
        estimations = pickle.load(file)
    print(parameters.shape, estimations.shape)
    print(estimations.shape)

    for i in range(parameters.shape[0]):
        if not ((parameters[i,dim:dim + dim**2] ==0.0) == (np.mean(estimations[i,:, dim:dim + dim**2] <2e-16, axis=0)>0.2)).all():

            print("Parameter: ", parameters[i,dim:dim + dim**2])
            print("Estimation: ", np.mean(estimations[i,:, dim:dim + dim**2] <2e-16, axis=0))
            print("Error: ", ((parameters[i,dim:dim + dim**2] ==0.0) == (np.mean(estimations[i,:, dim:dim + dim**2] <2e-16, axis=0)>0.2)).all())
            print("*"*100 )


    estimated_supports = np.mean(estimations[:,:, dim:dim + dim**2] <2e-16, axis=1)>0.2
    real_supports = parameters[:,dim:dim + dim**2] ==0.0
    res = np.all(estimated_supports == real_supports, axis=-1)
    print(res, np.mean(res))

    fig, ax = plt.subplots(figsize=(5.90666 * 2, 6))
    aux_list = [0.1 * i for i in range(2, 80)]
    y_list = []
    n_list = []
    for aux in aux_list:
        flag = parameters[:, -1] < aux
        parameters_aux = parameters[flag, :]
        estimations_aux = estimations[flag, :, :]

        estimated_supports_aux = np.mean(estimations_aux[:, :, dim:dim + dim ** 2] < 2e-16, axis=1) > 0.2
        real_supports_aux = parameters_aux[:, dim:dim + dim ** 2] == 0.0
        res_aux = np.all(estimated_supports_aux == real_supports_aux, axis=-1)

        y_list += [np.mean(res_aux)]
        n_list += [parameters_aux.shape[0]/repetitions]
    ax.plot(aux_list, y_list, label="Accuracy")
    ax.plot(aux_list, n_list, label="Proportion of total simulations.")
    ax.set_xlabel("$\\lambda_{max}$")
    plt.legend()
    for i,j in zip(aux_list, y_list):
        print(i,j)

    plt.savefig("graphics/accuracy_noise.pdf", format="pdf", bbox_inches="tight")

    plt.show()

