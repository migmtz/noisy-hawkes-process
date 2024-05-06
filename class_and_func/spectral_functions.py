import numpy as np
from scipy.linalg import inv, det


#############
# Periodogram
#############

def bartlett_periodogram(w, tList):
    T = tList[-1]
    t_aux = np.array(tList[1:-1])
    dt = np.sum(np.exp(- 2j * np.pi * w * t_aux))
    return ((1 / T) * dt * np.conj(dt)).real


def multivariate_periodogram(w, tList):
    max_time = tList[-1][0]
    dim = int(np.max(np.array(tList)[:, 1]))

    dimensional_times = [[t for t, i in tList if i == j] for j in range(1, dim + 1)]

    J_w = np.array([np.sum([np.exp(2j * np.pi * w * t) for t in i]) for i in dimensional_times]).reshape((dim, 1))
    return (1 / max_time) * J_w @ np.conj(J_w.T)


##################
# Spectral density
##################

def spectral_f_exp(w, theta):
    mu, alpha, beta = theta
    avg = mu / (1 - alpha)

    return avg * (1 + alpha * (beta**2) * (2 - alpha)/((beta*(1 - alpha))**2 + (2 * np.pi * w)**2))


def spectral_f_exp_noised(w, theta):
    mu, alpha, beta, lambda0 = theta
    f_val = spectral_f_exp(w, (mu, alpha, beta))

    return f_val + lambda0


def spectral_f_exp_grad(w, theta):
    mu, alpha, beta = theta
    avg = mu / (1 - alpha)
    D_ab = (beta * (1 - alpha)) ** 2 + (2 * np.pi * w)**2
    C_ab = 1 + alpha * (beta**2) * (2 - alpha)/D_ab
    f_val = avg * C_ab
    grad = np.zeros(3)
    grad[0] = C_ab / (1 - alpha)
    grad[1] = mu * C_ab * (2 * (beta**2) / D_ab + 1 / (1 - alpha)**2)
    grad[2] = 2 * avg * alpha * beta * (2 - alpha) * ((2 * np.pi * w)**2) / (D_ab**2)

    return f_val, grad[0], grad[1], grad[2]


def spectral_f_exp_noised_grad(w, theta):
    mu, alpha, beta, noise = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise, grad0, grad1, grad2, 1


# def spectral_f_exp_fixed(w, theta, idx, param):
#     print(theta[0:idx] , (param,) , theta[idx:])
#     theta_aux = theta[0:idx] + (param,) + theta[idx:]
#     res = np.array(spectral_f_exp_noised_grad(w, theta_aux))
#
#     return res[[i for i in range(5) if i!= (idx+1)]]


def spectral_f_exp_fixed(w, theta, idx, param):
    theta_aux = np.concatenate((theta[0:idx], [param], theta[idx:]))
    res = np.array(spectral_f_exp_noised_grad(w, theta_aux))

    return res[[i for i in range(5) if i!= (idx+1)]]


def multivariate_spectral_noised_density(w, theta):
    mu, alpha, beta, noise = theta
    dim = mu.shape[0]
    mean_matrix = np.identity(dim) * (inv(np.identity(dim) - alpha + 1e-12) @ mu)

    fourier_matrix = alpha * beta / (beta + 2j * np.pi * w)
    spectral_matrix = inv(np.identity(dim) - fourier_matrix)
    return np.conj(spectral_matrix) @ mean_matrix @ spectral_matrix.T + noise * np.identity(dim)


def multivariate_spectral_noised_single(w, theta):
    mu, alpha, beta, noise = theta
    # beta = np.array([beta_aux, 0]).reshape((2, 1))
    dim = mu.shape[0]
    m_1 = mu[0, 0]
    m_2 = mu[1, 0] + m_1 * alpha

    f_11 = m_1 + noise
    f_12 = m_1 * (alpha * beta) / (beta + 2j * np.pi * w)
    f_21 = m_1 * (alpha * beta) / (beta - 2j * np.pi * w)
    f_22 = m_1 * ((alpha * beta) ** 2) / (beta ** 2 + (2 * np.pi * w) ** 2) + m_2 + noise
    return np.array([[f_11, f_12], [f_21, f_22]])


def multivariate_spectral_noised_column(w, theta):
    mu, alpha_aux, beta, noise = theta
    alpha = np.hstack((alpha_aux.reshape((2, 1)), np.zeros((2, 1))))
    # beta = np.array([beta_aux, 0]).reshape((2, 1))
    dim = mu.shape[0]
    m_1 = mu[0, 0] / (1 - alpha[0, 0])
    m_2 = mu[1, 0] + m_1 * alpha[1, 0]

    aux_11 = (beta[0, 0] ** 2 + (2 * np.pi * w) ** 2) / (((1 - alpha[0, 0]) * beta[0, 0]) ** 2 + (2 * np.pi * w) ** 2)

    f_11 = m_1 * aux_11 + noise
    f_12 = m_1 * aux_11 * (alpha[1, 0] * beta[1, 0]) / (beta[1, 0] + 2j * np.pi * w)
    f_21 = m_1 * aux_11 * (alpha[1, 0] * beta[1, 0]) / (beta[1, 0] - 2j * np.pi * w)
    f_22 = m_1 * aux_11 * ((alpha[1, 0] * beta[1, 0]) ** 2) / (beta[1, 0] ** 2 + (2 * np.pi * w) ** 2) + m_2 + noise
    return np.array([[f_11, f_12], [f_21, f_22]])


def multivariate_spectral_noised_trinf(w, theta):
    mu, alpha_aux, beta, noise = theta
    alpha = np.array([[alpha_aux[0, 0], 0],
                      [alpha_aux[1, 0], alpha_aux[2, 0]]])
    # beta = np.array([beta_aux, 0]).reshape((2, 1))
    dim = mu.shape[0]
    m_1 = mu[0, 0] / (1 - alpha[0, 0])
    m_2 = mu[1, 0] / (1 - alpha[1, 1]) + m_1 * alpha[1, 0] / (1 - alpha[1, 1])

    aux_11 = (beta[0, 0] ** 2 + (2 * np.pi * w) ** 2) / (((1 - alpha[0, 0]) * beta[0, 0]) ** 2 + (2 * np.pi * w) ** 2)
    aux_22 = (beta[1, 0] ** 2 + (2 * np.pi * w) ** 2) / (((1 - alpha[1, 1]) * beta[1, 0]) ** 2 + (2 * np.pi * w) ** 2)

    f_11 = m_1 * aux_11 + noise
    f_12 = m_1 * aux_11 * (alpha[1, 0] * beta[1, 0]) / (beta[1, 0] * (1 - alpha[1, 1]) + 2j * np.pi * w)
    f_21 = m_1 * aux_11 * (alpha[1, 0] * beta[1, 0]) / (beta[1, 0] * (1 - alpha[1, 1]) - 2j * np.pi * w)
    f_22 = m_1 * aux_11 * ((alpha[1, 0] * beta[1, 0]) ** 2) / (
                (beta[1, 0] * (1 - alpha[1, 1])) ** 2 + (2 * np.pi * w) ** 2) + m_2 * aux_22 + noise
    return np.array([[f_11, f_12], [f_21, f_22]])



#########################
# Spectral log-likelihood
#########################


def spectral_log_likelihood(theta, f, M, periodogram):
    T = tList[-1]
    f_array = np.array([f(j / T, theta) for j in range(1, M+1)])
    pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array) * periodogram)

    return -pll


def spectral_log_likelihood_grad(theta, f, M, periodogram):
    T = tList[-1]
    f_array = np.array([f(j / T, theta) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
    print(-pll)

    return -pll, -pll_grad


def spectral_log_likelihood_grad_precomputed(theta, M, periodogram, max_time, idx_param, param):
    f_array = np.array([spectral_f_exp_fixed(2 * np.pi * j / max_time, theta, idx_param, param) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    pll = -(1/max_time) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/max_time) * np.sum(grad * aux.reshape(M, 1), axis=0)
    return -pll, -pll_grad


def spectral_multivariate_noised_ll(theta, M, periodogram, max_time):
    dim = int(np.sqrt(theta.shape[0]) - 1)
    theta_mid = theta[:-1]
    theta_aux = (theta_mid[:dim].reshape((dim, 1)), theta_mid[dim:-dim].reshape((dim, dim)), theta_mid[-dim:].reshape((dim, 1)), theta[-1])
    f_matrixes = [multivariate_spectral_noised_density(j/max_time, theta_aux) for j in range(1, M+1)]
    ll = np.sum([np.trace(inv(f_matrixes[i]) @ periodogram[i]) + np.log(det(f_matrixes[i])) for i in range(0, M)])
    return (1/max_time) * ll.real


def spectral_multivariate_noised_ll_single(theta, periodogram, K, max_time):
    # \alpha_{21} != 0
    dim = 2
    #theta_mid = theta[:-1]
    theta_aux = (np.array(theta[:dim]).reshape((dim, 1)),
                 theta[dim],
                 theta[dim+1],
                 theta[-1])
    f_matrixes = [multivariate_spectral_noised_single(j/max_time, theta_aux) for j in range(1, K+1)]
    ll = np.sum([np.trace(inv(f_matrixes[i]) @ periodogram[i]) + np.log(det(f_matrixes[i])) for i in range(0, K)])
    return (1/max_time) * ll


def spectral_multivariate_noised_ll_column(theta, periodogram, K, max_time):
    # \alpha_{i1} != 0
    dim = 2
    #theta_mid = theta[:-1]
    theta_aux = (np.array(theta[:dim]).reshape((dim, 1)),
                 np.array(theta[dim:dim+2]).reshape((dim, 1)),
                 np.array(theta[dim+2:-1]).reshape((dim,1)),
                 theta[-1])
    f_matrixes = [multivariate_spectral_noised_column(j/max_time, theta_aux) for j in range(1, K+1)]
    ll = np.sum([np.trace(inv(f_matrixes[i]) @ periodogram[i]) + np.log(det(f_matrixes[i])) for i in range(0, K)])
    return (1/max_time) * ll.real


def spectral_multivariate_noised_ll_trinf(theta, periodogram, K, max_time):
    # \alpha_{12} = 0
    dim = 2
    # theta_mid = theta[:-1]
    theta_aux = (np.array(theta[:dim]).reshape((dim, 1)),
                 np.array(theta[dim:dim + 3]).reshape((dim + 1, 1)),
                 np.array(theta[dim + 3:-1]).reshape((dim, 1)),
                 theta[-1])
    f_matrixes = [multivariate_spectral_noised_trinf(j / max_time, theta_aux) for j in range(1, K + 1)]
    ll = np.sum([np.trace(inv(f_matrixes[i]) @ periodogram[i]) + np.log(det(f_matrixes[i])) for i in range(0, K)])
    return (1 / max_time) * ll.real