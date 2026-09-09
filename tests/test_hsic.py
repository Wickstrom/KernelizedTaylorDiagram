import numpy as np
import pytest

from ktd import HSIC


@pytest.fixture
def data():
    rng = np.random.RandomState(123)
    X = rng.randn(100, 5)
    A = rng.rand(5, 5)
    return X, X @ A, rng


def test_same_data_normalized_score_is_one(data):
    X, _, _ = data
    hsic = HSIC(kernel="linear", center=True)
    hsic.fit(X, X)
    assert hsic.score(normalize=True) == pytest.approx(1.0)


def test_normalized_score_in_unit_interval_for_independent(data):
    X, _, rng = data
    Y = rng.randn(100, 5)
    hsic = HSIC(kernel="linear", center=True)
    hsic.fit(X, Y)
    score = hsic.score(normalize=True)
    assert 0.0 <= score < 0.1


def test_dependent_scores_higher_than_independent(data):
    X, Y_dep, rng = data
    Y_ind = rng.permutation(X)

    hsic = HSIC(kernel="linear", center=True)
    hsic.fit(X, Y_dep)
    dep_score = hsic.score(normalize=True)

    hsic = HSIC(kernel="linear", center=True)
    hsic.fit(X, Y_ind)
    ind_score = hsic.score(normalize=True)

    assert dep_score > ind_score


def test_rbf_kernel_with_fixed_gammas(data):
    X, Y, _ = data
    hsic = HSIC(kernel="rbf", gamma_X=1.0, gamma_Y=1.0, center=True)
    hsic.fit(X, Y)
    score = hsic.score(normalize=True)
    assert 0.0 <= score <= 1.0
    assert hsic.K_x_norm > 0
    assert hsic.K_y_norm > 0


def test_rbf_kernel_estimates_gamma_when_none():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)
    hsic = HSIC(kernel="rbf", center=True)
    hsic.fit(X, X)
    assert hsic.score(normalize=True) == pytest.approx(1.0)


def test_unnormalized_score_scaling(data):
    X, Y, _ = data
    hsic = HSIC(kernel="linear", center=True)
    hsic.fit(X, Y)

    bias = hsic.score(normalize=False)
    nobias = HSIC(kernel="linear", center=True, bias=False)
    nobias.fit(X, Y)
    nobias_score = nobias.score(normalize=False)

    assert bias == pytest.approx(hsic.hsic_value / hsic.n_samples**2)
    assert nobias_score == pytest.approx(
        nobias.hsic_value / (nobias.n_samples - 1) ** 2
    )


def test_mismatched_sample_sizes_raise():
    X = np.random.randn(50, 2)
    Y = np.random.randn(40, 2)
    hsic = HSIC(kernel="linear")
    with pytest.raises(ValueError, match="do not match"):
        hsic.fit(X, Y)


def test_subsample_preserves_row_pairing():
    """Regression test: X and Y rows must be subsampled jointly."""
    rng = np.random.RandomState(123)
    X = rng.randn(200, 3)
    Y = X + 1.0

    hsic = HSIC(kernel="rbf", gamma_X=1.0, gamma_Y=1.0, subsample=20, random_state=0)
    hsic.fit(X, Y)

    assert hsic.X_train_.shape == (20, 3)
    np.testing.assert_allclose(hsic.Y_train_, hsic.X_train_ + 1.0)


def test_callable_kernel():
    def my_kernel(X, Y=None, **kwargs):
        return X @ Y.T if Y is not None else X @ X.T

    rng = np.random.RandomState(0)
    X = rng.randn(30, 4)
    hsic = HSIC(kernel=my_kernel, center=True)
    hsic.fit(X, X)
    assert hsic.score(normalize=True) == pytest.approx(1.0)
