"""Alternative QML model implementations for the two-moons benchmark."""

from .chen_sun_vqc import ChenSUNVQCModel
from .projected_model import ProjectedTrainableModel
from .simonetti_hybrid import SimonettiHybridModel
from .souza_sqqnn import SouzaSQQNNModel

__all__ = [
    "ChenSUNVQCModel",
    "ProjectedTrainableModel",
    "SimonettiHybridModel",
    "SouzaSQQNNModel",
]
