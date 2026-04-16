"""Shared optimizer dispatchers."""

from __future__ import annotations

from ._classification_impl import run_optimizer as run_optimizer
from ._vqe_impl import build_initial_params, run_optimizer as run_vqe_optimizer

__all__ = ["build_initial_params", "run_optimizer", "run_vqe_optimizer"]
