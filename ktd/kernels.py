"""Kernel bandwidth estimation utilities.

Adapted from the ``pysim`` package by J. Emmanuel Johnson
(https://github.com/jejjohnson/pysim, MIT license).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_array, check_random_state

__all__ = [
    "estimate_gamma",
    "estimate_sigma",
    "gamma_to_sigma",
    "sigma_to_gamma",
]


def gamma_to_sigma(gamma: float) -> float:
    """Transform the gamma parameter into sigma using the relationship

    .. math::
        \\sigma = \\frac{1}{\\sqrt{2 \\gamma}}
    """
    return 1 / np.sqrt(2 * gamma)


def sigma_to_gamma(sigma: float) -> float:
    """Transform the sigma parameter into gamma using the relationship

    .. math::
        \\gamma = \\frac{1}{2 \\sigma^2}
    """
    return 1 / (2 * sigma**2)


def estimate_gamma(
    X: np.ndarray,
    subsample: int | None = None,
    method: str = "median",
    percent: float | None = 0.15,
    scale: float = 1.0,
    random_state: int | None = None,
) -> float:
    """Estimate the gamma parameter for an RBF kernel from data.

    Parameters
    ----------
    X : array, shape (n_samples, d_dimensions)
        The data matrix to be estimated.
    subsample : int, optional
        Number of samples to subsample before estimating.
    method : str, default='median'
        The method used to estimate the sigma ('mean', 'median',
        'silverman' or 'scott').
    percent : float, default=0.15
        The kth percentage of distances chosen.
    scale : float, default=1.0
        Optional scale factor applied to the estimated sigma.
    random_state : int, optional
        Seed for the subsampling.

    Returns
    -------
    gamma : float
        The estimated gamma value.
    """
    sigma = estimate_sigma(
        X=X,
        subsample=subsample,
        method=method,
        percent=percent,
        scale=scale,
        random_state=random_state,
    )
    return sigma_to_gamma(sigma)


def estimate_sigma(
    X: np.ndarray,
    subsample: int | None = None,
    method: str = "median",
    percent: float | None = 0.15,
    scale: float = 1.0,
    random_state: int | None = None,
) -> float:
    """Provide a reasonable estimate of the sigma value for an RBF kernel.

    Parameters
    ----------
    X : array, shape (n_samples, d_dimensions)
        The data matrix to be estimated.
    subsample : int, optional
        Number of samples to subsample before estimating.
    method : str, default='median'
        The method used to estimate the sigma:

        * 'mean' - mean pairwise distance
        * 'median' - median pairwise distance
        * 'silverman' - Silverman's rule of thumb
        * 'scott' - Scott's rule of thumb

    percent : float, default=0.15
        The kth percentage of distances chosen (used with
        'mean'/'median').
    scale : float, default=1.0
        Optional scale factor applied to the estimated sigma.
    random_state : int, optional
        Seed for the subsampling.

    Returns
    -------
    sigma : float
        The estimated sigma value.

    References
    ----------
    Original MATLAB function: https://goo.gl/xYoJce
    """
    X = check_array(X, ensure_2d=True)
    rng = check_random_state(random_state)

    if subsample is not None:
        X = rng.permutation(X)[:subsample, :]

    n_samples, d_dimensions = X.shape

    if method == "mean":
        if percent is None:
            sigma = np.mean(pdist(X))
        else:
            kth_sample = int(percent * n_samples)
            sigma = np.mean(np.sort(squareform(pdist(X)))[:, kth_sample])

    elif method == "median":
        if percent is None:
            sigma = np.median(pdist(X))
        else:
            kth_sample = int(percent * n_samples)
            sigma = np.median(np.sort(squareform(pdist(X)))[:, kth_sample])

    elif method == "silverman":
        sigma = np.power(
            n_samples * (d_dimensions + 2.0) / 4.0, -1.0 / (d_dimensions + 4)
        )

    elif method == "scott":
        sigma = np.power(n_samples, -1.0 / (d_dimensions + 4))

    else:
        raise ValueError(f"Unrecognized method: '{method}'.")

    if scale is not None:
        sigma *= scale

    return sigma
