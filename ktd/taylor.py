"""Taylor diagram plotting.

The diagram implementation is based on Yannick Copin's original
implementation (https://gist.github.com/ycopin/3342888) and a modified
version from a StackExchange code review thread
(https://codereview.stackexchange.com/questions/82919/).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder

__all__ = ["TaylorDiagram"]


class TaylorDiagram:
    """Creates a Taylor diagram.

    In the kernelized Taylor diagram, the reference point is the
    (logarithm of the) Frobenius norm of the reference kernel matrix
    and the angle of each sample is the kernel alignment score.

    Parameters
    ----------
    ref_point : float
        The reference point (radius) for the diagram.
    fig : matplotlib.figure.Figure, optional
        The figure to add the diagram to. A new figure is created if
        not provided.
    subplot : int or tuple, default=111
        The subplot specification.
    extend_angle : bool, default=False
        Whether to extend the angle range to cover negative
        correlations.
    corr_labels : array, optional
        The correlation labels to draw on the angular axis.
    ref_range : tuple, default=(0, 10)
        The range of the radial axis, expressed relative to the
        reference point. The upper limit is
        ``ref_range[1] / 100 * ref_point + ref_point``.
    ref_label : str, default='Reference Point'
        Label for the reference point.
    angle_label : str, default='Correlation'
        Label for the angular axis.
    var_label : str, default='Standard Deviation'
        Label for the radial axis.

    References
    ----------
    Original Implementation:
        - Yannick Copin
        - https://gist.github.com/ycopin/3342888
    Modified Implementation:
        - https://codereview.stackexchange.com/questions/82919/
    """

    def __init__(
        self,
        ref_point: float,
        fig: plt.Figure | None = None,
        subplot=111,
        extend_angle: bool = False,
        corr_labels: np.ndarray | None = None,
        ref_range: tuple[float, float] = (0, 10),
        ref_label: str = "Reference Point",
        angle_label: str = "Correlation",
        var_label: str = "Standard Deviation",
    ) -> None:
        self.ref_point = ref_point
        self.ref_label = ref_label
        self.angle_label = angle_label
        self.var_label = var_label
        self.extend_angle = extend_angle

        # correlation labels
        if corr_labels is None:
            corr_labels = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

        if extend_angle:
            # extend to negative correlations
            self.tmax = np.pi
            corr_labels = np.concatenate((-corr_labels[:0:-1], corr_labels))
        else:
            # limit to positive correlations
            self.tmax = np.pi / 2.0

        # init figure
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # radial range
        self.smin = ref_range[0] * ref_point
        self.smax = (ref_range[1] / 100) * ref_point + ref_point

        corr_ticks = np.arccos(corr_labels)
        gl1 = grid_finder.FixedLocator(corr_ticks)
        tf1 = grid_finder.DictFormatter(dict(zip(corr_ticks, map(str, corr_labels))))

        ghelper = floating_axes.GridHelperCurveLinear(
            aux_trans=PolarAxes.PolarTransform(),
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        ax = floating_axes.FloatingSubplot(fig, subplot, grid_helper=ghelper)
        fig.add_subplot(ax)
        self.graph_axes = ax
        self.polar_axes = ax.get_aux_axes(PolarAxes.PolarTransform())

        self.sample_points = []
        self.reset_axes()

    def add_reference_point(self, ref_point: float, *args, **kwargs) -> None:
        """Add the reference point to the diagram."""
        line = self.polar_axes.plot([0], ref_point, *args, **kwargs)
        self.sample_points.append(line[0])

    def add_reference_line(self, ref_point: float, *args, **kwargs) -> None:
        """Add the horizontal reference line at ``ref_point``."""
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + ref_point
        self.polar_axes.plot(t, r, *args, **kwargs)

    def add_point(self, var_point: float, corr_point: float, *args, **kwargs) -> None:
        """Add a single sample point to the diagram."""
        line = self.polar_axes.plot(np.arccos(corr_point), var_point, *args, **kwargs)
        self.sample_points.append(line[0])

    def add_scatter(
        self, var_points: np.ndarray, corr_points: np.ndarray, *args, **kwargs
    ) -> None:
        """Add a collection of sample points to the diagram."""
        pts = self.polar_axes.scatter(
            np.arccos(corr_points), var_points, *args, **kwargs
        )
        self.sample_points.append(pts)

    def add_grid(self, *args, **kwargs) -> None:
        """Add a grid to the diagram."""
        self.graph_axes.grid(*args, **kwargs)

    def add_contours(self, ref_point: float, levels: int = 4, **kwargs) -> None:
        """Add RMSD-style contours around the reference point."""
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )
        dist = np.sqrt(ref_point**2 + rs**2 - 2 * ref_point * rs * np.cos(ts))
        self.contours = self.polar_axes.contour(ts, rs, dist, levels=levels, **kwargs)

    def add_legend(self, fig: plt.Figure | None = None, *args, **kwargs) -> None:
        """Add a legend with all the sample points added so far."""
        if fig is None:
            fig = plt.gcf()
        fig.legend(
            self.sample_points,
            [p.get_label() for p in self.sample_points],
            *args,
            **kwargs,
        )

    def reset_axes(self) -> None:
        """Reset the axes configuration."""
        self._setup_angle_axes()
        self._setup_xaxis()
        self._setup_yaxis()

    def reset_axes_labels(
        self, angle_label: str = "Correlation", var_label: str = "Variance"
    ) -> None:
        """Reset only the axis labels."""
        self.graph_axes.axis["left"].label.set_text(var_label)
        self.graph_axes.axis["top"].label.set_text(angle_label)

    def _setup_angle_axes(self) -> None:
        self.graph_axes.axis["top"].set_axis_direction("bottom")
        self.graph_axes.axis["top"].toggle(ticklabels=True, label=True)
        self.graph_axes.axis["top"].major_ticklabels.set_axis_direction("top")
        self.graph_axes.axis["top"].label.set_axis_direction("top")
        self.graph_axes.axis["top"].label.set_text(self.angle_label)

    def _setup_xaxis(self) -> None:
        self.graph_axes.axis["left"].set_axis_direction("bottom")
        self.graph_axes.axis["left"].label.set_text(self.var_label)

    def _setup_yaxis(self) -> None:
        self.graph_axes.axis["right"].set_axis_direction("top")
        self.graph_axes.axis["right"].toggle(ticklabels=True)
        self.graph_axes.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if self.extend_angle else "left"
        )
        self.graph_axes.axis["bottom"].toggle(ticklabels=False, label=False)
