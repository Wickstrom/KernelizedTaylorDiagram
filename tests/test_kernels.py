import numpy as np
import pytest

from ktd import estimate_gamma, estimate_sigma, gamma_to_sigma, sigma_to_gamma


def test_sigma_gamma_roundtrip():
    rng = np.random.RandomState(42)
    gammas = rng.uniform(0.01, 100.0, size=50)
    restored = sigma_to_gamma(gamma_to_sigma(gammas))
    np.testing.assert_allclose(restored, gammas)


def test_estimate_sigma_median_matches_manual():
    X = np.random.RandomState(0).randn(50, 3)
    from scipy.spatial.distance import pdist

    expected = np.median(pdist(X))
    assert estimate_sigma(X, percent=None) == pytest.approx(expected)


def test_estimate_sigma_mean():
    X = np.random.RandomState(0).randn(30, 2)
    sigma = estimate_sigma(X, method="mean", percent=None)
    assert sigma > 0


@pytest.mark.parametrize("method", ["mean", "median", "silverman", "scott"])
def test_estimate_sigma_methods_positive(method):
    X = np.random.RandomState(0).randn(20, 2)
    assert estimate_sigma(X, method=method) > 0


def test_estimate_sigma_invalid_method_raises():
    X = np.random.RandomState(0).randn(10, 2)
    with pytest.raises(ValueError, match="Unrecognized method"):
        estimate_sigma(X, method="does-not-exist")


def test_estimate_sigma_scale():
    X = np.random.RandomState(0).randn(20, 2)
    base = estimate_sigma(X, percent=None)
    scaled = estimate_sigma(X, percent=None, scale=2.0)
    assert scaled == pytest.approx(2.0 * base)


def test_estimate_gamma_is_inverse_of_sigma():
    X = np.random.RandomState(0).randn(20, 2)
    sigma = estimate_sigma(X, percent=None)
    gamma = estimate_gamma(X, percent=None)
    assert gamma == pytest.approx(sigma_to_gamma(sigma))


def test_estimate_sigma_subsample_random_state_reproducible():
    X = np.random.RandomState(0).randn(100, 2)
    a = estimate_sigma(X, subsample=10, random_state=123)
    b = estimate_sigma(X, subsample=10, random_state=123)
    assert a == b
