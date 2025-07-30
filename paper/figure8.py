import pandas as pd
import numpy as np
from class_and_func.spectral_functions import estimate_spectral_density
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()


titles = ["$f_{11}$", "$f_{22}$"]


if __name__ == "__main__":
    df = pd.read_csv('../real_data_application/spk_mouse22.csv', sep=',', header=0, index_col=0)

    max_time = 725
    partition_size = 725//5
    partition_list = [partition_size * i for i in range(0, max_time // partition_size +1)]
    print(partition_list)

    dim = 2
    repetitions = 5
    K_func = lambda x : int(x)
    #K_func = lambda x: int(x * np.log(x))

    data = [df.values[(df.values[:, 2] == i) * (df.values[:, 1] > 4.0), 0:2] for i in range(1,6)]
    variates_idx = {5.0:1, 6.0:2}
    data = [np.array([(t - i * max_time, variates_idx[m]) for (t,m) in data[i]]) for i in range(0,5)]
    print(data[0][:,0])

    partitioned_data = []

    for i in range(len(partition_list) - 1):
        for parasited_times in data:
            flag1 = parasited_times[:, 0] <= partition_list[i + 1]
            flag2 = partition_list[i] < parasited_times[:, 0]

            aux_parasited = parasited_times[flag1 * flag2, :]
            partitioned_data += [aux_parasited]

    print(len(partitioned_data))
    # Data count
    print("# of points per repetition:", [len(u) for u in partitioned_data])

    x_freq, final_ITx = estimate_spectral_density(
        partitioned_data, partition_size)

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 4.5))

    for k in range(2):
        ax[k].plot(x_freq, final_ITx[:, k, k], label="Periodogram", alpha=1.0)
        ax[k].set_title(titles[k])
        ax[k].set_xlabel("$\\nu$")

    ax[0].legend()

    plt.savefig("graphics/non_parametric_periodogram.pdf", format="pdf", bbox_inches="tight")
    plt.show()