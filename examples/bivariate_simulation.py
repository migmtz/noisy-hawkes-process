from class_and_func.simulation_exponential_hawkes import multivariate_exponential_hawkes
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()


if __name__ == "__main__":
    # Parameters
    mu = np.array([[1.0],
                   [1.0]])
    alpha = np.array([[0.5, 0.0],
                      [0.4, 0.4]])
    beta = np.array([[1.0],
                     [1.3]])

    print("Spectral radius:", np.max(np.abs(np.linalg.eig(alpha)[0])))

    max_jumps = 1000
    burn_in = -10.0

    noise = 0.5
    theta = np.concatenate((mu.ravel(), alpha.ravel(), beta.ravel()))
    theta = np.append(theta[theta > 0.0], noise)

    np.random.seed(0)
    hp = multivariate_exponential_hawkes(mu, alpha, beta, max_jumps=max_jumps, burn_in=burn_in)
    hp.simulate()
    hp_times = hp.timestamps

    hp.plot_intensity(where=50)

    plt.show()