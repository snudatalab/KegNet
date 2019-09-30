"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

This code is based on
https://github.com/CasvandenBogaard/VBMF/blob/master/VBMF.py
"""

from __future__ import division

import numpy as np
from scipy import optimize


# noinspection PyPep8Naming,SpellCheckingInspection
def EVBMF(tensor, sigma2=None, H=None):
    """
    Implementation of the analytical solution to Empirical Variational Bayes
    Matrix Factorization.

    This function can be used to calculate the analytical solution to empirical
    VBMF. This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix
    factorization."

    Notes
    -----
    If sigma2 is unspecified, it is estimated by minimizing the free energy.
    If H is unspecified, it is set to the smallest of the sides of the input Y.

    Attributes
    ----------
    tensor : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.

    sigma2 : int or None (default=None)
        Variance of the noise on Y.

    H : int or None (default = None)
        Maximum rank of the factorized matrices.

    Returns
    -------
    U : numpy-array
        Left-singular vectors.

    S : numpy-array
        Diagonal matrix of singular values.

    V : numpy-array
        Right-singular vectors.

    post : dictionary
        Dictionary containing the computed posterior values.


    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of
    fully-observed variational Bayesian matrix factorization." Journal of
    Machine Learning Research 14.Jan (2013): 1-37.

    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by
    variational Bayesian PCA." Advances in Neural Information Processing
    Systems. 2012.
    """
    tensor = tensor.cpu().numpy()
    L, M = tensor.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    U, s, V = np.linalg.svd(tensor)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.
    if H < L:
        residual = np.sum(np.sum(tensor ** 2) - np.sum(s ** 2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        upper_bound = (np.sum(s ** 2) + residual) / (L * M)
        lower_bound = np.max(
            [s[eH_ub + 1] ** 2 / (M * xubar),
             np.mean(s[eH_ub + 1:] ** 2) / M])

        scale = 1.  # /lower_bound
        s = s * np.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale

        sigma2_opt = optimize.minimize_scalar(
            EVBsigma2,
            args=(L, M, s, residual, xubar),
            bounds=[lower_bound, upper_bound],
            method='Bounded')
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
    pos = np.sum(s > threshold)

    # Formula (15) from [2]
    d = np.multiply(
        s[:pos] / 2, 1 - np.divide((L + M) * sigma2, s[:pos] ** 2) +
        np.sqrt((1 - np.divide((L + M) * sigma2, s[:pos] ** 2)) ** 2 -
                4 * L * M * sigma2 ** 2 / s[:pos] ** 4))

    # Computation of the posterior
    post = {'ma': np.zeros(H),
            'mb': np.zeros(H),
            'sa2': np.zeros(H),
            'sb2': np.zeros(H),
            'cacb': np.zeros(H)}

    tau = np.multiply(d, s[:pos]) / (M * sigma2)
    delta = np.multiply(np.sqrt(np.divide(M * d, L * s[:pos])), 1 + alpha / tau)

    post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    post['sa2'][:pos] = np.divide(sigma2 * delta, s[:pos])
    post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos]) / (L * M))
    post['sigma2'] = sigma2
    post['F'] = 0.5 * (
            L * M * np.log(2 * np.pi * sigma2) +
            (residual + np.sum(s ** 2)) / sigma2 +
            np.sum(M * np.log(tau + 1) + L * np.log(tau / alpha + 1) - M * tau))

    return U[:, :pos], np.diag(d), V[:, :pos], post


# noinspection PyPep8Naming,SpellCheckingInspection
def EVBsigma2(sigma2, L, M, s, residual, xubar):
    """
    Function for EVBMF function (minimization objective function).
    """

    # noinspection PyShadowingNames
    def tau(x, alpha):
        y = np.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha)
        return 0.5 * (x - (1 + alpha) + y)

    H = len(s)

    alpha = L / M
    x = s ** 2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1 + 1, z1)))
    term4 = alpha * np.sum(np.log(tau_z1 / alpha + 1))

    obj1 = term1 + term2 + term3 + term4
    obj2 = residual / (M * sigma2)
    obj3 = (L - H) * np.log(sigma2)

    return obj1 + obj2 + obj3
