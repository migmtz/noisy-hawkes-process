import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
from class_and_func.colormaps import get_continuous_cmap
import warnings


class exp_thinning_hawkes(object):
    """
    Univariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    """

    def __init__(self, mu, alpha, beta, t=0.0, max_jumps=None, max_time=None):
        """
        Parameters
        ----------
        mu : float
            Baseline constant intensity.
        alpha : float
            Interaction factor.
        beta : float
            Decay factor.
        t : float, optional
            Initial time. The default is 0.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.

        Attributes
        ----------
        t_0 : float
            Initial time provided at initialization.
        timestamps : list of float
            List of simulated events. It includes the initial time t_0.
        intensity_jumps : list of float
            List of intensity at each simulated jump. It includes the baseline intensity lambda_0.
        aux : float
            Parameter used in simulation.
        simulated : bool
            Parameter that marks if a process has been already been simulated, or if its event times have been initialized.
        """
        self.alpha = alpha
        self.beta = beta
        self.t_0 = t
        self.t = t
        self.mu = mu
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.timestamps = [t]
        self.intensity_jumps = [mu]
        self.aux = 0
        self.simulated = False

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm.

        Works with both self-exciting and self-regulating processes.

        To launch simulation either self.max_jumps or self.max_time must be other than None, so the algorithm knows when to stop.
        """
        if not self.simulated:
            if self.max_jumps is not None and self.max_time is None:
                self.simulate_jumps()
            elif self.max_time is not None and self.max_jumps is None:
                self.simulate_time()
            else:
                print("Either max_jumps or max_time must be given.")
            self.simulated = True

        else:
            print("Process already simulated")

    def simulate_jumps(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0

        candidate_intensity = self.mu

        while flag < self.max_jumps:

            upper_intensity = max(self.mu,
                                  self.mu + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1])))

            self.t += np.random.exponential(1 / upper_intensity)
            candidate_intensity = self.mu + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1]))

            if upper_intensity * np.random.uniform() <= candidate_intensity:
                self.timestamps += [self.t]
                self.intensity_jumps += [candidate_intensity + self.alpha * self.beta]
                self.aux = candidate_intensity - self.mu + self.alpha * self.beta
                flag += 1

        self.max_time = self.timestamps[-1]
        # We have to add a "self.max_time = self.timestamps[-1] at the end so plot_intensity works correctly"

    def simulate_time(self):
        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
        flag = self.t < self.max_time

        while flag:
            upper_intensity = max(self.mu,
                                  self.mu + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1])))

            self.t += np.random.exponential(1 / upper_intensity)
            candidate_intensity = self.mu + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1]))
            #print("cand", self.t, candidate_intensity)

            flag = self.t < self.max_time

            if upper_intensity * np.random.uniform() <= candidate_intensity and flag:
                self.timestamps += [self.t]
                self.intensity_jumps += [candidate_intensity + self.alpha * self.beta]
                self.aux = self.aux * np.exp(-self.beta * (self.t - self.timestamps[-2])) + self.alpha * self.beta
        # self.timestamps += [self.max_time]

    def plot_intensity(self, ax=None, plot_N=True):
        """
        Plot intensity function. If plot_N is True, plots also step function N([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        plot_N : bool, optional.
            Whether we plot the step function N or not.
        """
        if not self.simulated:
            print("Simulate first")

        else:
            if plot_N:
                if ax is None:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                elif isinstance(ax[0], matplotlib.axes.Axes):
                    ax1, ax2 = ax
                else:
                    return "ax must be a (2,1) axes"
            else:
                if ax is None:
                    fig, ax1 = plt.subplots()
                elif isinstance(ax, matplotlib.axes.Axes):
                    ax1 = ax
                else:
                    return "ax must be an instance of an axes"

            self.timestamps.append(self.max_time)

            times = np.array([self.timestamps[0], self.timestamps[1]])
            intensities = np.array([self.mu, self.mu])
            step = 0.01
            for i, lambda_k in enumerate(self.intensity_jumps):
                if i != 0:
                    T_k = self.timestamps[i]
                    nb_step = np.maximum(100, np.floor((self.timestamps[i + 1] - T_k) / step))
                    aux_times = np.linspace(T_k, self.timestamps[i + 1], int(nb_step))
                    times = np.append(times, aux_times)
                    intensities = np.append(intensities, self.mu + (lambda_k - self.mu) * np.exp(
                        -self.beta * (aux_times - T_k)))

            ax1.plot([0, self.max_time], [0, 0], c='k', alpha=0.5)
            # if self.alpha < 0:
            # ax1.plot(times, intensities, label="Underlying intensity", c="#1f77b4")
            ax1.plot(times, np.maximum(intensities, 0), label="Conditional intensity", c='r')
            ax1.legend()
            ax1.grid()
            if plot_N:
                ax2.step(self.timestamps, np.append(np.arange(0, len(self.timestamps) - 1), len(self.timestamps) - 2),
                         where="post", label="$N(t)$")
                ax2.legend()
                ax2.grid()
            self.timestamps.pop()

    def set_time_intensity(self, timestamps):

        """
        Method to initialize a Hawkes process with a given list of timestamps.

        It computes the corresponding intensity with respect to the parameters given at initialization.

        Parameters
        ----------
        timestamps : list of float
            Imposed jump times. Intensity is adjusted to this list of times. Must be ordered list of times.
            It is best if obtained by simulating from another instance of Hawkes process.

        """

        if not self.simulated:
            self.timestamps = timestamps
            self.max_time = timestamps[-1]

            intensities = [self.mu]
            for k in range(1, len(timestamps)):
                intensities += [self.mu + (intensities[-1] - self.mu) * np.exp(
                    -self.beta * (timestamps[k] - timestamps[k - 1])) + self.alpha * self.beta]
            self.intensity_jumps = intensities
            self.simulated = True

        else:
            print("Already simulated")

    def compensator_transform(self, plot=None, exclude_values=0):
        """
        Obtains transformed times for use of goodness-of-fit tests.

        Transformation obtained through time change theorem.

        Parameters
        ----------
        plot : .axes.Axes, optional.
            If None, then it just obtains the transformed times, otherwise plot the Q-Q plot. The default is None
        exclude_values : int, optional.
            If 0 then takes all transformed points in account during plot. Otherwise, excludes first 'exclude_values'
            values from Q-Q plot. The default is 0.
        """

        if not self.simulated:
            print("Simulate first")

        else:

            T_k = self.timestamps[1]

            compensator_k = self.mu * (T_k - self.t_0)

            self.timestamps_transformed = [compensator_k]
            self.intervals_transformed = [compensator_k]

            for k in range(2, len(self.timestamps)):

                lambda_k = self.intensity_jumps[k - 1]
                tau_star = self.timestamps[k] - self.timestamps[k - 1]
                if lambda_k >= 0:
                    C_k = lambda_k - self.mu
                else:
                    C_k = -self.mu
                    tau_star -= (np.log(-(lambda_k - self.mu)) - np.log(self.mu)) / self.beta

                compensator_k = self.mu * tau_star + (C_k / self.beta) * (1 - np.exp(-self.beta * tau_star))

                self.timestamps_transformed += [self.timestamps_transformed[-1] + compensator_k]
                self.intervals_transformed += [compensator_k]

            if plot is not None:
                stats.probplot(self.intervals_transformed[exclude_values:], dist=stats.expon, plot=plot)


class multivariate_exponential_hawkes(object):
    """
    Multivariate Hawkes process with exponential kernel. No events nor initial conditions considered.
    """

    def __init__(self, mu, alpha, beta, max_jumps=None, max_time=None, burn_in=0.0):
        """

        Parameters
        ----------
        mu : array_like
            Baseline intensity vector. mu.shape[0] must coincide with shapes for alpha and beta.
        alpha : array_like
            Interaction factors matrix. Must be a square array with alpha.shape[0] coinciding with mu and beta.
        beta : array_like
            Decay factor matrix. Must be either an array. When corresponding to decay for each process i, it must
            be of shape (number_of_process, 1), or a square array. beta.shape[0] must coincide with mu and alpha.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.
        burn_in : float, optional
            Burn-in period for stationary simulation.

        Attributes
        ----------
        nb_processes : int
            Number of dimensions.
        timestamps : list of tuple (float, int)
            List of simulated events and their marks.
        intensity_jumps : array of float
            Array containing all intensities at each jump. It includes the baseline intensities mu.
        simulated : bool
            Parameter that marks if a process has been already been simulated,
            or if its event times have been initialized.

        """

        # We must begin by verifying that the process is a point process. In other words, that the number of
        # points in any bounded interval is a.s. finite. For this, we have to verify that the spectral radius of
        # the matrix alpha/beta (term by term) is <1.

        beta_radius = np.copy(beta)
        beta_radius[beta_radius == 0] = 1
        self.spectral_radius = np.max(np.abs(np.linalg.eig(np.abs(alpha))[0]))

        if self.spectral_radius >= 1:
            # raise ValueError("Spectral radius is %s, which makes the process unstable." % (spectral_radius))
            warnings.warn("Spectral radius is %s, which may make the process unstable." % (self.spectral_radius),RuntimeWarning)
        self.mu = mu.reshape((alpha.shape[0], 1))
        self.alpha = alpha
        if beta.shape[1] != alpha.shape[0]: # Initialisation of beta if constant by process
            self.beta = np.repeat(beta, alpha.shape[0], axis=1)
        else:
            self.beta = beta
        self.max_jumps = max_jumps
        self.max_time = max_time

        self.before_origin_time = 0.0
        self.aux_before = 0.0 * self.alpha
        self.at_origin_intensity = self.mu

        self.burn_in = -np.abs(burn_in)

        self.nb_processes = self.mu.shape[0]
        self.count = np.zeros(self.nb_processes, dtype=int)

        self.timestamps = [(burn_in, 0)]
        self.intensity_jumps = np.copy(mu)

        self.simulated = False

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm. Upper bound obtained by the positive-part process.

        Works with both self-exciting and self-regulating processes.

        To launch simulation either self.max_jumps or self.max_time must be other than None, so the algorithm knows when to stop.
        """
        if not self.simulated:
            if self.max_jumps is not None and self.max_time is None:
                self.simulate_jumps()
            elif self.max_time is not None and self.max_jumps is None:
                if self.spectral_radius >= 1.0:
                    raise ValueError("Spectral radius is %s, simulation with max_time may not end. Prefer providing max_jumps." % (self.spectral_radius))
                self.simulate_time()
            else:
                print("Either max_jumps or max_time must be given.")
            self.simulated = True
            #self.complete_times = self.timestamps
            self.timestamps = [(0.0, 0)] + [(t, m) for t,m in self.timestamps if t > 0.0]

        else:
            print("Process already simulated")

    def simulate_jumps(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0
        t = self.burn_in

        auxiliary_alpha = np.where(self.alpha > 0, self.alpha, 0)
        auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
        auxiliary_intensity = np.copy(self.mu)

        ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

        while flag < self.max_jumps:

            upper_intensity = np.sum(auxiliary_intensity)

            previous_t = t
            t += np.random.exponential(1 / upper_intensity)
            if t > 0.0 > previous_t:
                self.before_origin_time = previous_t
                self.aux_before = np.multiply(ij_intensity, np.exp(-self.beta * (0 - previous_t)))
                self.at_origin_intensity = self.mu + np.sum(self.aux_before, axis=1, keepdims=True)

            # ij_intensity = np.multiply(ij_intensity, np.exp(-self.beta * (t - self.timestamps[-1][0])))
            ij_intensity = np.multiply(ij_intensity, np.exp(-self.beta * (t - previous_t)))
            candidate_intensities = self.mu + np.sum(ij_intensity, axis=1, keepdims=True)
            pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
            type_event = np.random.multinomial(1, np.concatenate((pos_candidate.squeeze(axis=1), np.array([0.0])))).argmax()
            if type_event < self.nb_processes:
                self.timestamps += [(t, type_event + 1)]
                ij_intensity[:, type_event] += self.alpha[:, type_event] * self.beta[:, type_event]
                self.intensity_jumps = np.c_[
                    self.intensity_jumps, self.mu + np.sum(ij_intensity, axis=1, keepdims=True)]

                auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.beta * (t - self.timestamps[-2][0])))
                auxiliary_ij[:, type_event] += auxiliary_alpha[:, type_event] * self.beta[:, type_event]
                auxiliary_intensity = self.mu + np.sum(auxiliary_ij, axis=1, keepdims=True)

                flag += 1

                self.count[type_event] += 1

        self.max_time = self.timestamps[-1][0]
        # Important to add the max_time for plotting and being consistent.
        self.timestamps += [(self.max_time, 0)]
        #self.timestamps = [(t,m) for t,m in self.timestamps()]

    def simulate_time(self):
        """
        Simulation is done for a window [0, T] (T = self.max_time) is attained.
        """
        t = self.burn_in
        flag = t < self.max_time

        auxiliary_alpha = np.where(self.alpha > 0, self.alpha, 0)
        auxiliary_ij = np.zeros((self.nb_processes, self.nb_processes))
        auxiliary_intensity = np.copy(self.mu)

        ij_intensity = np.zeros((self.nb_processes, self.nb_processes))

        while flag:

            upper_intensity = np.sum(auxiliary_intensity)

            previous_t = t
            t += np.random.exponential(1 / upper_intensity)
            if t > 0.0 > previous_t:
                self.before_origin_time = previous_t
                self.aux_before = np.multiply(ij_intensity, np.exp(-self.beta * (0 - previous_t)))
                self.at_origin_intensity = self.mu + np.sum(self.aux_before, axis=1, keepdims=True)
            #print("cand", t, upper_intensity)

            # ij_intensity = np.multiply(ij_intensity, np.exp(-self.beta * (t - self.timestamps[-1][0])))
            ij_intensity = np.multiply(ij_intensity, np.exp(-self.beta * (t - previous_t)))
            candidate_intensities = self.mu + np.sum(ij_intensity, axis=1, keepdims=True)
            pos_candidate = np.maximum(candidate_intensities, 0) / upper_intensity
            #print((pos_candidate.squeeze(axis=1), pos_candidate.squeeze(), np.array([0.0])))
            type_event = np.random.multinomial(1,
                                               np.concatenate((pos_candidate.squeeze(axis=1), np.array([0.0])))).argmax()
            flag = t < self.max_time
            if type_event < self.nb_processes and flag:
                self.timestamps += [(t, type_event + 1)]
                ij_intensity[:, type_event] += self.alpha[:, type_event] * self.beta[:, type_event]
                self.intensity_jumps = np.c_[
                    self.intensity_jumps, self.mu + np.sum(ij_intensity, axis=1, keepdims=True)]

                auxiliary_ij = np.multiply(auxiliary_ij, np.exp(-self.beta * (t - self.timestamps[-2][0])))
                auxiliary_ij[:, type_event] += auxiliary_alpha[:, type_event] * self.beta[:, type_event]
                auxiliary_intensity = self.mu + np.sum(auxiliary_ij, axis=1, keepdims=True)

                self.count[type_event] += 1

        self.timestamps += [(self.max_time, 0)]


    def plot_intensity(self, ax=None, plot_N=True, where=10):
        """
        Plot intensity function. If plot_N is True, plots also step functions N^i([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be array of shape (2,K) if plot_N = True, or (K,) if plot_N = False
        plot_N : bool, optional.
            Whether we plot the step function N^i or not.
        """

        if not self.simulated:
            print("Simulate first")

        else:
            plt.rcParams['axes.grid'] = True
            if plot_N:
                jumps_plot = [[0] for i in range(self.nb_processes)]
                if ax is None:
                    fig, ax = plt.subplots(2, self.nb_processes, sharex=True)
                elif isinstance(ax[0,0], matplotlib.axes.Axes):
                    pass
                else:
                    return "ax is the wrong shape. It should be (2, number of processes+1)"
                ax1 = ax[0]
                ax2 = ax[1]
                if self.nb_processes == 1:
                    ax1 = [ax1]
                    ax2 = [ax2]
            else:
                if ax is None:
                    fig, ax1 = plt.subplots(1, self.nb_processes)
                elif isinstance(ax, matplotlib.axes.Axes) or isinstance(ax, np.ndarray):
                    ax1 = ax
                else:
                    return "ax is the wrong shape. It should be (number of processes+1,)"
                if self.nb_processes == 1:
                    ax1 = [ax1]


            ij_intensity = self.aux_before

            step = 100

            if self.before_origin_time < 0.0:
                func = lambda x: self.mu + np.matmul(
                    np.multiply(ij_intensity, np.exp(-self.beta * (x))),
                    np.ones((self.nb_processes, 1)))

                # On enregistre la division de temps et les sauts
                interval_t = np.linspace(0.0, self.timestamps[1][0], step)
                times = interval_t.tolist()
                intensities = np.array(list(map(func, interval_t))).squeeze(axis=-1).T

            else:
                times = [0, self.timestamps[1][0]]
                intensities = np.array([[self.mu[i, 0], self.mu[i, 0]] for i in range(self.nb_processes)])

            # print("here", self.timestamps[len(self.timestamps)])
            #print("where",len(self.timestamps[1:where]), where)
            for i in range(1, len(self.timestamps[1:where])):
                # On commence par mettre à jour la matrice lambda^{ij}
                ij_intensity = np.multiply(ij_intensity,
                                           np.exp(-self.beta * (self.timestamps[i][0] - self.timestamps[i - 1][0])))
                # On enregistre le saut d'intensité de l'évenement, pour son type.
                ij_intensity[:, self.timestamps[i][1]-1] += self.alpha[:, self.timestamps[i][1]-1] * self.beta[:, self.timestamps[i][1]-1]

                # On définit la fonction à tracer entre T_n et T_{n+1}
                func = lambda x: self.mu + np.matmul(
                    np.multiply(ij_intensity, np.exp(-self.beta * (x - self.timestamps[i][0]))),
                                np.ones((self.nb_processes, 1)))

                # On enregistre la division de temps et les sauts
                interval_t = np.linspace(self.timestamps[i][0], self.timestamps[i + 1][0], step)
                times += interval_t.tolist()
                intensities = np.concatenate((intensities, np.array(list(map(func, interval_t))).squeeze(axis=-1).T ), axis=1)
                if plot_N:
                    jumps_plot[self.timestamps[i][1]-1] += [self.timestamps[i][0] for t in range(2)]

            for i in range(self.nb_processes):
                ax1[i].plot(times, intensities[i], label="Underlying intensity", c="#1f77b4", linestyle="--")
                ax1[i].plot(times, np.maximum(intensities[i], 0), label="Conditional intensity", c='r')
                # ax1[i].plot([i for i,j in self.timestamps[:-1]], self.intensity_jumps[i,:], c='k', alpha=0.5)

            ax1[0].legend()

            if plot_N:
                for i in range(self.nb_processes):
                    jumps_plot[i] += [times[-1]]
                    ax2[i].plot(jumps_plot[i], [t for t in range(len(jumps_plot[i])//2) for j in range(2)], c="r", label="Process #%s"%(i+1), zorder=10)
                    # ax2[i].set_ylim(ax2[i].get_ylim())
                    for j in range(self.nb_processes):
                        if j != i:
                            ax2[j].plot(jumps_plot[i], [t for t in range(len(jumps_plot[i])//2) for j in range(2)], c="#1f77b4", alpha=0.5, zorder=5)

                    ax2[i].legend()



    def plot_heatmap(self, ax=None):
        """
        This function allows to observe the heatmap where each cell {ij} corresponds to the value {alpha/beta} from that interaction

        Parameters
        ----------
        ax : .axes.Axes, optional.
            If None, method will generate own ax.
            Otherwise, will use given ax.
        """
        import seaborn as sns

        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax = ax
        beta_heat = np.copy(self.beta)
        beta_heat[beta_heat == 0] = 1
        heat_matrix = self.alpha

        hex_list = ['#FF3333', '#FFFFFF', '#33FF49']

        ax = sns.heatmap(heat_matrix, cmap=get_continuous_cmap(hex_list), center=0, ax=ax, annot=True)
