"""Adam optimizers for classification and VQE benchmarks."""

from ._classification_impl import train_adam
from ._vqe_impl import train_adam as train_vqe_adam

__all__ = ["train_adam", "train_vqe_adam"]
