"""Shared classification datasets for QML benchmarks."""

from .iris import load_iris_binary
from .parity import generate_parity_4bit, generate_parity_4bit_unique_split
from .perez_salinas import (
    PROBLEM_DEFAULT_SAMPLES,
    available_perez_salinas_problems,
    generate_perez_salinas_dataset,
    perez_salinas_4q8l_preset,
    perez_salinas_benchmark_preset,
    perez_salinas_problem_num_classes,
)
from .two_moons import generate_two_moons

__all__ = [
    "PROBLEM_DEFAULT_SAMPLES",
    "available_perez_salinas_problems",
    "generate_parity_4bit",
    "generate_parity_4bit_unique_split",
    "generate_perez_salinas_dataset",
    "generate_two_moons",
    "load_iris_binary",
    "perez_salinas_4q8l_preset",
    "perez_salinas_benchmark_preset",
    "perez_salinas_problem_num_classes",
]
