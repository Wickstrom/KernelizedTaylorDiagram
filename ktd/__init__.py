"""Kernelized Taylor Diagram.

A Python implementation of the kernelized Taylor diagram, a graphical
framework for visualizing similarities between data populations:

    Wickstrøm, K., Johnson, J. E., Løkse, S., Camps-Valls, G.,
    Mikalsen, K. Ø., Kampffmeyer, M., & Jenssen, R. (2022).
    "The Kernelized Taylor Diagram." arXiv:2205.08864.
    https://arxiv.org/abs/2205.08864

Parts of this package are adapted from the ``pysim`` package by
J. Emmanuel Johnson (https://github.com/jejjohnson/pysim, MIT license).
"""

from .hsic import HSIC
from .kernels import estimate_gamma, estimate_sigma, gamma_to_sigma, sigma_to_gamma
from .taylor import TaylorDiagram

__version__ = "0.1.0"

__all__ = [
    "HSIC",
    "TaylorDiagram",
    "estimate_gamma",
    "estimate_sigma",
    "gamma_to_sigma",
    "sigma_to_gamma",
]
