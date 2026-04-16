"""Alternative QML model implementations for the two-moons benchmark."""

from .chen_sun_vqc import ChenSUNVQCModel
from .parity_rot_classifier import ParityRotClassifierModel
from .perez_salinas_reuploading import PerezSalinasReuploadingModel
from .projected_model import ProjectedTrainableModel
from .simonetti_hybrid import SimonettiHybridModel
from .souza_sqqnn import SouzaSQQNNModel

__all__ = [
    "ChenSUNVQCModel",
    "ParityRotClassifierModel",
    "PerezSalinasReuploadingModel",
    "ProjectedTrainableModel",
    "SimonettiHybridModel",
    "SouzaSQQNNModel",
]
