"""Hilbert-Schmidt Independence Criterion (HSIC).

Adapted from the ``pysim`` package by J. Emmanuel Johnson
(https://github.com/jejjohnson/pysim, MIT license).
"""

from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_array, check_random_state

from .kernels import estimate_gamma

__all__ = ["HSIC"]


class HSIC(BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC).

    A method for measuring (in)dependence between two variables.
    Variants found in the literature:

    * HSIC - centered, unnormalized
    * KA  - uncentered, normalized
    * CKA - centered, normalized

    Parameters
    ----------
    gamma_X : float, optional
        Gamma parameter for the X kernel. If None, it is estimated
        from the data (RBF-like kernels only).
    gamma_Y : float, optional
        Gamma parameter for the Y kernel. If None, it is estimated
        from the data (RBF-like kernels only).
    kernel : str or callable, default='linear'
        Kernel mapping used internally. A callable should accept two
        arguments and return a kernel matrix. Set to 'precomputed' to
        pass precomputed kernel matrices to :meth:`fit`.
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : dict, optional
        Additional parameters (keyword arguments) for the kernel
        function passed as a callable object.
    center : bool, default=True
        Whether to center the kernel matrices after construction.
    subsample : int, optional
        Number of samples to subsample (rows are sampled jointly from
        X and Y to preserve pairing).
    bias : bool, default=True
        Normalization used for the unnormalized HSIC estimate.
        Only relevant when calling :meth:`score` with
        ``normalize=False``.
    random_state : int, optional
        Seed used for subsampling.

    Attributes
    ----------
    hsic_value : float
        The raw (summed) HSIC value computed during :meth:`fit`.
    K_x_norm, K_y_norm : float
        Frobenius norms of the (optionally centered) kernel matrices.

    Example
    -------
    >>> import numpy as np
    >>> from ktd import HSIC
    >>> rng = np.random.RandomState(123)
    >>> X = rng.randn(100, 5)
    >>> Y = X @ rng.rand(5, 5)
    >>> hsic_clf = HSIC(center=True, kernel='linear')
    >>> hsic_clf.fit(X, Y)
    >>> cka_score = hsic_clf.score(normalize=True)
    """

    def __init__(
        self,
        gamma_X: float | None = None,
        gamma_Y: float | None = None,
        kernel: Callable | str = "linear",
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict | None = None,
        center: bool = True,
        subsample: int | None = None,
        bias: bool = True,
        random_state: int | None = None,
    ):
        self.gamma_X = gamma_X
        self.gamma_Y = gamma_Y
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.center = center
        self.subsample = subsample
        self.bias = bias
        self.random_state = random_state
        self.rng = check_random_state(random_state)

    def fit(self, X, Y):
        """Compute the kernel matrices and the HSIC value.

        Parameters
        ----------
        X : array, shape (n_samples, d_x)
            First data matrix.
        Y : array, shape (n_samples, d_y)
            Second data matrix (same number of samples as X).

        Returns
        -------
        self
        """
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Samples of X ({X.shape[0]}) and Y ({Y.shape[0]}) do not match"
            )

        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]

        # Subsample jointly to preserve the row pairing between X and Y
        if self.subsample is not None:
            indices = self.rng.permutation(self.n_samples)[: self.subsample]
            X = X[indices, :]
            Y = Y[indices, :]

        self.X_train_ = X
        self.Y_train_ = Y

        # Calculate the kernel matrices
        K_x = self.compute_kernel(X, gamma=self.gamma_X)
        K_y = self.compute_kernel(Y, gamma=self.gamma_Y)

        # Center the kernels
        if self.center:
            K_x = KernelCenterer().fit_transform(K_x)
            K_y = KernelCenterer().fit_transform(K_y)

        # Compute the HSIC value
        self.hsic_value = np.sum(K_x * K_y)

        # Kernel magnitudes
        self.K_x_norm = np.linalg.norm(K_x)
        self.K_y_norm = np.linalg.norm(K_y)

        return self

    def compute_kernel(self, X, Y=None, gamma=None):
        """Compute the kernel matrix for X (and optionally Y)."""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            if gamma is None:
                gamma = estimate_gamma(X)
            params = {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def score(self, X=None, y=None, normalize: bool = True) -> float:
        """Return the alignment score between the two kernel matrices.

        With ``normalize=True`` this returns the (centered) kernel
        alignment, ``<K_x, K_y> / (||K_x|| * ||K_y||)``, in [0, 1].
        With ``normalize=False`` it returns the bias-corrected HSIC
        estimate.

        Parameters
        ----------
        X, y : ignored
            Present for scikit-learn API compatibility.
        normalize : bool, default=True
            Whether to return the normalized alignment score.
        """
        if normalize:
            return self.hsic_value / self.K_x_norm / self.K_y_norm

        if self.bias:
            self.hsic_bias = 1 / (self.n_samples**2)
        else:
            self.hsic_bias = 1 / (self.n_samples - 1) ** 2

        return self.hsic_bias * self.hsic_value
