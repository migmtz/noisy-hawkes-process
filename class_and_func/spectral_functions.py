import numpy as np
from scipy.linalg import inv, det
import finufft


#############
# Periodogram
#############

def bartlett_periodogram(w, tList): # To use if finufft not available.
    T = tList[-1]
    t_aux = np.array(tList[1:-1])
    dt = np.sum(np.exp(- 2j * np.pi * w * t_aux))
    return ((1 / T) * dt * np.conj(dt)).real


def multivariate_periodogram(w, tList): # To use if finufft not available.
    max_time = tList[-1][0]
    dim = int(np.max(np.array(tList)[:, 1]))

    dimensional_times = [[t for t, i in tList if i == j] for j in range(1, dim + 1)]

    J_w = np.array([np.sum([np.exp(-2j * np.pi * w * t) for t in i]) for i in dimensional_times]).reshape((dim, 1))
    return (1 / max_time) * J_w @ np.conj(J_w.T)


def fast_multi_periodogram(K, tList, max_time, precision=1e-9):
    dim = int(np.max(np.array(tList)[:, 1]))
    dimensional_times = [[t for t, i in tList if i == j] for j in range(1, dim + 1)]

    # put K for w=0
    aux = np.array([finufft.nufft1d1(2 * np.pi * np.array(x) / max_time, np.ones(len(x)) + 0j, n_modes=2 * K + 1,
                                     isign=-1, eps=1e-9)[K + 1:] for x in dimensional_times])
    aux = (aux.T)[:, :, np.newaxis]
    aux = (aux @ np.transpose(np.conj(aux), axes=(0, 2, 1))) / max_time

    return aux


def fast_multi_periodogram_window(tList, max_time, window, precision=1e-9, debiased=True):
    # Computes the periodogram in a window [-window, window] with a step of 1/max_time.
    dim = int(np.max(np.array(tList)[:, 1]))
    K = int(np.ceil(window * max_time))

    dimensional_times = [[t for t, i in tList if i == j] for j in range(1, dim + 1)]

    # put K for w=0
    aux = np.array([finufft.nufft1d1(2 * np.pi * np.array(x) / max_time, np.ones(len(x)) + 0j, n_modes=2 * K + 1,
                                     isign=-1, eps=1e-9) for x in dimensional_times])
    aux = (aux.T)[:, :, np.newaxis]
    aux = (aux @ np.transpose(np.conj(aux), axes=(0, 2, 1))) / max_time
    if debiased:
        aux[K] = 0 # Remove value at v = 0.

    return aux


#############
# Auxiliary
#############


def ei(size, index):
    e = np.zeros((size))
    e[index] = 1.0
    return e


##################
# Univariate
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
    #print(theta)
    mu, alpha, beta, noise = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise, grad0, grad1, grad2, 1


def spectral_f_exp_fixed(w, theta, idx, param):
    #print(theta, param)
    theta_aux = np.concatenate((theta[0:idx], param, theta[idx:]))
    res = np.array(spectral_f_exp_noised_grad(w, theta_aux))

    return res[[i for i in range(5) if i!= (idx+1)]]

################
# Multivariate #
################


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


def spectral_w_mask(theta, w):
    # Compute f(v_k) for noised process
    mu0, alpha0, beta0, noise0 = theta

    dim = mu0.shape[0]
    a = inv(np.identity(dim) - alpha0)

    mean_matrix = np.identity(dim) * (a @ mu0)

    fourier_matrix = alpha0 * beta0 / (beta0 + 2j * np.pi * w)
    spectral_matrix = inv(np.identity(dim) - fourier_matrix)

    f_theta_unnoised = (spectral_matrix) @ mean_matrix @ np.conj(spectral_matrix.T)
    f_theta = f_theta_unnoised + noise0 * np.identity(dim)

    return f_theta.real


def spectral_unnoised_w_mask(theta, w, periodogramw): # The mask allows to set values to 0.
    # Compute f(v_k) for Hawkes process
    mu0, alpha0, beta0 = theta

    dim = mu0.shape[0]
    if np.max(np.abs(np.linalg.eig(alpha0)[0])) > 1.0:
        return np.array([1e10])# + [0.0]*dim + [1e10]*(dim**2) + [0.0]*dim )
    a = inv(np.identity(dim) - alpha0)

    mean_matrix = np.identity(dim) * (a @ mu0)

    fourier_matrix = alpha0 * beta0 / (beta0 + 2j * np.pi * w)
    aux = np.identity(dim) - fourier_matrix
    spectral_matrix = inv(aux)

    f_theta = (spectral_matrix) @ mean_matrix @ np.conj(spectral_matrix.T)
    f_inv = np.conj(aux.T) @ inv(mean_matrix) @ aux

    ll = np.log(det(f_theta)) + np.trace(f_inv @ periodogramw)

    return ll


def grad_spectral_w_mask(theta, w, periodogramw, mask):
    # The mask allows to set values to 0.
    # Compute term (v_k) in spectral-loglikelihood for noised process with gradient
    mu0, alpha0, beta0, noise0 = theta

    dim = mu0.shape[0]
    a = inv(np.identity(dim) - alpha0)

    mean_matrix = np.identity(dim) * (a @ mu0)

    fourier_matrix = alpha0 * beta0 / (beta0 + 2j * np.pi * w)
    spectral_matrix = inv(np.identity(dim) - fourier_matrix)

    f_theta_unnoised = (spectral_matrix) @ mean_matrix @ np.conj(spectral_matrix.T)
    f_theta = f_theta_unnoised + noise0 * np.identity(dim)
    f_inv = inv(f_theta)

    ll = np.log(det(f_theta)) + np.trace(f_inv @ periodogramw)

    aux_dbeta = alpha0 * (2j * np.pi * w) * np.repeat(1 / (beta0 + 2j * np.pi * w) ** 2, dim, axis=1)

    dmu = a @ np.array([ei(((dim, dim)), i) for i in range(dim)]) * np.array([np.identity(dim)] * dim)
    dbeta = aux_dbeta * np.array([ei((dim, dim), i) for i in range(dim)])

    dij = mask * np.array([[ei(dim, i)[:, np.newaxis] * ei(dim, j)[np.newaxis, :] for j in range(dim)] for i in range(dim)])
    dalpha_cent = a @ dij @ a @ mu0
    dalpha_cent = dalpha_cent * np.array([[np.identity(dim)] * dim] * dim)
    dalpha_bord = dij * beta0 / (beta0 + 2j * np.pi * w)

    dmu = spectral_matrix @ dmu @ np.conj(spectral_matrix.T)
    dalpha = (spectral_matrix @ dalpha_bord @ f_theta_unnoised) + (
                f_theta_unnoised @ np.transpose(np.conj(dalpha_bord), axes=(0, 1, 3, 2)) @ np.conj(spectral_matrix.T))
    dalpha += spectral_matrix @ dalpha_cent @ np.conj(spectral_matrix.T)
    dbeta = (spectral_matrix @ dbeta @ f_theta_unnoised) + (
                f_theta_unnoised @ np.transpose(np.conj(dbeta), axes=(0, 2, 1)) @ np.conj(spectral_matrix.T))
    dnoise = np.identity(dim)
    aux_det = f_inv.T

    aux_trace_mu = (aux_det.T) @ dmu @ (aux_det.T)
    aux_trace_alpha = (aux_det.T) @ dalpha @ (aux_det.T)
    aux_trace_beta = (aux_det.T) @ dbeta @ (aux_det.T)
    aux_trace_noise = (aux_det.T) @ dnoise @ (aux_det.T)

    dmu = np.sum(aux_det * dmu, axis=(1,2)) - np.sum((periodogramw.T) * aux_trace_mu,
                                                                        axis=(1,2))
    dalpha = np.sum(aux_det * dalpha, axis=(2, 3)) - np.sum((periodogramw.T) * aux_trace_alpha, axis=(2, 3))
    dbeta = np.sum(aux_det * dbeta, axis=(1,2)) - np.sum((periodogramw.T) * aux_trace_beta,
                                                                            axis=(1,2))
    dnoise = np.sum(aux_det * dnoise) - np.sum((periodogramw.T) * aux_trace_noise)

    grad_final = np.concatenate((dmu.real.ravel(), dalpha.real.ravel(), dbeta.real.ravel(), np.array([dnoise.real])))

    return np.concatenate((np.array([ll.real]), grad_final))


def grad_spectral_unnoised_w_mask(theta, w, periodogramw, mask):
    # The mask allows to set values to 0.
    # Compute term (v_k) in spectral-loglikelihood for Hawkes process with gradient
    mu0, alpha0, beta0 = theta

    dim = mu0.shape[0]

    if np.max(np.abs(np.linalg.eig(alpha0)[0])) > 1.0:
        return np.array([1e10] + [0.0]*dim + [1e10]*(dim**2) + [0.0]*dim )
    a = inv(np.identity(dim) - alpha0)

    mean_matrix = np.identity(dim) * (a @ mu0)

    fourier_matrix = alpha0 * beta0 / (beta0 + 2j * np.pi * w)
    aux = np.identity(dim) - fourier_matrix
    spectral_matrix = inv(aux)

    f_theta = (spectral_matrix) @ mean_matrix @ np.conj(spectral_matrix.T)
    f_inv = np.conj(aux.T) @ inv(mean_matrix) @ aux

    ll = np.log(det(f_theta) + 1e-16) + np.trace(f_inv @ periodogramw)

    aux_dbeta = alpha0 * (2j * np.pi * w) * np.repeat(1 / (beta0 + 2j * np.pi * w) ** 2, dim, axis=1)

    dmu = a @ np.array([ei(((dim, dim)), i) for i in range(dim)]) * np.array([np.identity(dim)] * dim)
    dbeta = aux_dbeta * np.array([ei((dim, dim), i) for i in range(dim)])

    dij = mask * np.array([[ei(dim, i)[:, np.newaxis] * ei(dim, j)[np.newaxis, :] for j in range(dim)] for i in range(dim)])
    dalpha_cent = a @ dij @ a @ mu0
    dalpha_cent = dalpha_cent * np.array([[np.identity(dim)] * dim] * dim)
    dalpha_bord = dij * beta0 / (beta0 + 2j * np.pi * w)

    dmu = spectral_matrix @ dmu @ np.conj(spectral_matrix.T)
    dalpha = (spectral_matrix @ dalpha_bord @ f_theta) + (
                f_theta @ np.transpose(np.conj(dalpha_bord), axes=(0, 1, 3, 2)) @ np.conj(spectral_matrix.T))
    dalpha += spectral_matrix @ dalpha_cent @ np.conj(spectral_matrix.T)
    dbeta = (spectral_matrix @ dbeta @ f_theta) + (
                f_theta @ np.transpose(np.conj(dbeta), axes=(0, 2, 1)) @ np.conj(spectral_matrix.T))

    aux_det = f_inv.T

    aux_trace_mu = (f_inv) @ dmu @ (f_inv)
    aux_trace_alpha = (f_inv) @ dalpha @ (f_inv)
    aux_trace_beta = (f_inv) @ dbeta @ (f_inv)

    dmu = np.sum(aux_det * dmu, axis=(1,2)) - np.sum((periodogramw.T) * aux_trace_mu,
                                                                        axis=(1,2))
    dalpha = np.sum(aux_det * dalpha, axis=(2, 3)) - np.sum((periodogramw.T) * aux_trace_alpha, axis=(2, 3))
    dbeta = np.sum(aux_det * dbeta, axis=(1,2)) - np.sum((periodogramw.T) * aux_trace_beta,
                                                                            axis=(1,2))

    grad_final = np.concatenate((dmu.real.ravel(), dalpha.real.ravel(), dbeta.real.ravel()))
    return np.concatenate((np.array([ll.real]), grad_final))

#########################
# Spectral log-likelihood
#########################


# def spectral_log_likelihood_grad(theta, f, M, periodogram):
#     # Allows to compute the spectral log_likelihood with gradient for univariate process.
#     T = tList[-1]
#     f_array = np.array([f(j / T, theta) for j in range(1, M+1)])
#     f_val, grad = f_array[:, 0], f_array[:, 1:]
#     pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
#     aux = (1/f_val) * (1 - (1/f_val) * periodogram)
#     pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
#     print(-pll)
#
#     return -pll, -pll_grad


def spectral_log_likelihood_grad_precomputed(theta, M, periodogram, max_time, idx_param, param):
    # Compute spectral-loglikelihood with gradient.
    periodo = periodogram.squeeze()

    f_array = np.array([spectral_f_exp_fixed(j / max_time, theta, idx_param, param) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    pll = -(1/max_time) * np.sum(np.log(f_val) + (1/f_val) * periodo)
    aux = (1/f_val) * (1 - (1/f_val) * periodo)
    pll_grad = -(1/max_time) * np.sum(grad * aux.reshape(M, 1), axis=0)
    return -pll.real, -pll_grad.real


# def spectral_multivariate_noised_ll_single(theta, periodogram, K, max_time):
#     # \alpha_{21} != 0
#     dim = 2
#     #theta_mid = theta[:-1]
#     theta_aux = (np.array(theta[:dim]).reshape((dim, 1)),
#                  theta[dim],
#                  theta[dim+1],
#                  theta[-1])
#     f_matrixes = [multivariate_spectral_noised_single(j/max_time, theta_aux) for j in range(1, K+1)]
#     ll = np.sum([np.trace(inv(f_matrixes[i]) @ periodogram[i]) + np.log(det(f_matrixes[i])) for i in range(0, K)])
#     return (1/max_time) * ll


def grad_ll_unnoised_mask(theta, periodogram, K, max_time, mask=None):
    # Mask is a matrix of the same shape as alpha with True for known non-null entries.
    # Used to estimate in the reduced multivariate noised model.
    dim = (periodogram[0]).shape[0]

    if mask is not None:
        mask_aux = mask

        list_aux = list(theta)
        param_mask = np.concatenate(([True] * dim, mask.ravel(), mask.any(axis=1)))
        theta_mask = np.zeros((dim * (2 + dim)))
        theta_mask[-dim - 1:] = 1

        true_indices = np.where(param_mask)[0]
        theta_mask[true_indices] = list_aux[:len(true_indices)]

    else:
        mask_aux = np.ones((dim, dim))
        param_mask = np.array([True] * (dim * (2 + dim)))
        theta_mask = theta

    theta_aux = (
    theta_mask[:dim].reshape((dim, 1)), theta_mask[dim:-dim].reshape((dim, dim)), theta_mask[-dim:].reshape((dim, 1)))
    if np.max(np.abs(np.linalg.eig(theta_aux[1])[0])) > 1.0:
        return np.array([1e10]), np.array([0.0] * dim + [1e10] * (dim ** 2) + [0.0] * (dim + 1))

    ll = np.sum([grad_spectral_unnoised_w_mask(theta_aux, j / max_time, periodogram[j - 1], mask_aux) for j in range(1, K + 1)],
                axis=0)
    ll /= max_time

    return (ll[0], ll[1:][param_mask])


def ll_unnoised_mask(theta, periodogram, K, max_time, mask=None):
    # Mask is a matrix of the same shape as alpha with True for known non-null entries.
    # Used to estimate in the reduced multivariate Hawkes model.
    dim = (periodogram[0]).shape[0]

    if mask is not None:
        mask_aux = mask

        list_aux = list(theta)
        param_mask = np.concatenate(([True] * dim, mask.ravel(), mask.any(axis=1)))
        theta_mask = np.zeros((dim * (2 + dim)))
        theta_mask[-dim - 1:] = 1

        true_indices = np.where(param_mask)[0]
        theta_mask[true_indices] = list_aux[:len(true_indices)]

    else:
        mask_aux = np.ones((dim, dim))
        param_mask = np.array([True] * (dim * (2 + dim)))
        theta_mask = theta

    theta_aux = (
    theta_mask[:dim].reshape((dim, 1)), theta_mask[dim:-dim].reshape((dim, dim)), theta_mask[-dim:].reshape((dim, 1)))
    if np.max(np.abs(np.linalg.eig(theta_aux[1])[0])) > 1.0:
        return np.array([1e10])

    ll = np.sum([spectral_unnoised_w_mask(theta_aux, j / max_time, periodogram[j - 1], mask_aux) for j in range(1, K + 1)],
                axis=0)
    ll /= max_time

    return ll


def grad_ll_mask(theta, periodogram, K, max_time, mask=None):
    dim = (periodogram[0]).shape[0]

    if mask is not None:
        mask_aux = mask

        list_aux = list(theta)
        param_mask = np.concatenate(([True] * dim, mask.ravel(), mask.any(axis=1), [True]))
        theta_mask = np.zeros((dim * (2 + dim) + 1))
        theta_mask[-dim - 1:-1] = 1

        true_indices = np.where(param_mask)[0]
        theta_mask[true_indices] = list_aux[:len(true_indices)]

    else:
        mask_aux = np.ones((dim, dim))
        param_mask = np.array([True] * (dim * (2 + dim) + 1))
        theta_mask = theta

    theta_mid = theta_mask[:-1]
    theta_aux = (
    theta_mid[:dim].reshape((dim, 1)), theta_mid[dim:-dim].reshape((dim, dim)), theta_mid[-dim:].reshape((dim, 1)),
    theta_mask[-1])
    if np.max(np.abs(np.linalg.eig(theta_aux[1])[0])) > 1.0:
        return np.array([1e10]),np.array([0.0]*dim + [1e10]*(dim**2) + [0.0]*(dim+1))

    ll = np.sum([grad_spectral_w_mask(theta_aux, j / max_time, periodogram[j - 1], mask_aux) for j in range(1, K + 1)],
                axis=0)
    ll /= max_time

    return (ll[0], ll[1:][param_mask])