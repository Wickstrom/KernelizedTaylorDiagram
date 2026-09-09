import matplotlib.pyplot as plt
import numpy as np
import pytest

from ktd import TaylorDiagram


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def test_diagram_creation():
    td = TaylorDiagram(ref_point=1.0)
    assert td.tmax == np.pi / 2
    assert td.smin == 0.0
    assert td.smax == pytest.approx(1.1)
    assert td.graph_axes is not None
    assert td.polar_axes is not None


def test_diagram_extended_angle():
    td = TaylorDiagram(ref_point=2.0, extend_angle=True)
    assert td.tmax == np.pi


def test_diagram_custom_ref_range():
    td = TaylorDiagram(ref_point=1.0, ref_range=(0, 30))
    assert td.smax == pytest.approx(1.3)


def test_add_elements_and_save(tmp_path, rng):
    td = TaylorDiagram(ref_point=1.5)

    td.add_reference_point(1.5, color="black", marker=".", markersize=20, label="ref")
    td.add_reference_line(1.5, color="black", linestyle="--")
    td.add_grid()
    td.add_contours(1.5, levels=3, colors="gray")

    n = 5
    stds = rng.uniform(0.5, 1.5, size=n)
    corrs = rng.uniform(0.2, 0.99, size=n)
    td.add_scatter(stds, corrs, c="blue", s=50)

    out = tmp_path / "taylor.png"
    plt.savefig(out)
    assert out.exists() and out.stat().st_size > 0

    assert len(td.sample_points) == 2  # reference point + scatter


def test_add_single_point(tmp_path, rng):
    td = TaylorDiagram(ref_point=1.0)
    td.add_point(0.8, 0.9, c="red", marker="x")
    out = tmp_path / "point.png"
    plt.savefig(out)
    assert out.exists()


def test_ref_point_is_stored():
    # regression guard: the diagram must remember its reference point
    td = TaylorDiagram(ref_point=3.21)
    assert td.ref_point == pytest.approx(3.21)
