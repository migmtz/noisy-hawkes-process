# Spectral estimation for Noisy Hawkes processes

Source code for [Spectral analysis for noisy Hawkes processes inference](TBD) [[1]](#1). 

This code implements parametric estimation of Hawkes process when noised by an independent, homogeneous Poisson process.
Data is assumed to be the superposition of these two point processes, which corresponds to the ordered union of event times. We assume that both processes are indistinguishable, in other words, there is no way of knowing whether a point comes from the Hawkes process or the Poisson process.

This leverages spectral analysis of point processes and what is known as the Bartlett spectrum. The Bartlett spectrum of the superposition of two independent processes corresponds to the sum of respective spectra.

Estimations are obtained by maximisation of a spectral equivalent of the log-likelihood (in this code, we implement the minimisation of the opposite quantity through the minimize method of Scipy).

## Functions in ```spectral_functions.py```
The functions ```spectral_log_likelihood``` and ```spectral_log_likelihood_grad``` are implemented as general expressions dependent on a function (parameter f) which computes the spectral density of a point process. 

The ```periodogram``` is a quantity that can be precomputed in order to accelerate the estimation, as this quantity is not dependent on the type of point processes. All implementation of the spectral log-likelihood take directly the periodogram as a parameter.

The implemented spectral density functions correspond to the exponential kernel function $h(t) = \alpha \beta \exp^{-\beta t}$ for $t>0$.

## Example

We include two examples of application:
* ```univariate_noisy_estimation.py``` for an estimation in the univariate setting, which requires fixing one parameter to assure identifiability of the model.
* ```bivariate_noisy_estimation.py``` for an estimation in the bivaraite setting. (Proven) Identifiable settings assume that certain interactions are null. Function
```spectral_multivariate_noised_ll``` can be used to estimate the null interactions before implementing one of the identifiable cases.

## Dependencies

This code was implemented using Python 3.8.5 and needs Numpy, Matplotlib and Scipy.

Pickle package must be used to used the precomputed estimations presented in the paper.
Jupyter notebooks use multiprocess package.

## Installation

Copy all files in the current working directory.

## Author

Miguel Alejandro Martinez Herrera

## References

<a id="1">[1]</a>
A. Bonnet, M. Martinez Herrera, M. Sangnier, Spectral analysis for noisy Hawkes processes inference. arXiv:TBD
