"""
File: hydrogen_wavefunction.py
Description: Vectorized computational core for hydrogenic bound-state eigenfunctions.

Model computations:
   - Normalized radial functions with stable log-gamma normalization & complex spherical harmonics.
   - Stationary-state wavefunction on an x-z plane grid (y=0).
   - Reduced-mass Bohr radius and electron-nucleus reduced mass.

Model assumptions:
   - Non-relativistic, point nucleus, Schrodinger hydrogenic Hamiltonian with Coulomb potential.
   - No spin/fine-structure, external fields, or finite-nuclear-size effects.
   - Shapes broadcast; R is real-valued, Y and psi are complex.

Author: Sebastian Mag (adapted)
Repository: https://github.com/ssebastianmag/hydrogen-wavefunctions
"""

from typing import Optional, Literal

import numpy as np
import scipy.special as sp
from scipy.constants import physical_constants, m_e, m_p


def radial_wavefunction_Rnl(
    n: int,
    l: int,
    r: np.ndarray,
    Z: int = 1,
    use_reduced_mass: bool = True,
    M: Optional[float] = None,
):
    """Normalized hydrogenic radial wavefunction R_{n,l}(r)."""
    if not (n >= 1 and 0 <= l <= n - 1):
        raise ValueError("Quantum numbers (n,l) must satisfy n >= 1 and 0 <= l <= n-1")

    mu = reduced_electron_nucleus_mass(Z, M) if use_reduced_mass else m_e
    a_mu = reduced_bohr_radius(mu)

    rho = 2.0 * Z * r / (n * a_mu)
    L = sp.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)

    # Stable normalization prefactor using log-gamma
    log_pref = 1.5 * np.log(2.0 * Z / (n * a_mu))
    log_pref += 0.5 * (
        sp.gammaln(n - l) - (np.log(2.0 * n) + sp.gammaln(n + l + 1))
    )
    pref = np.exp(log_pref)
    R = pref * np.exp(-rho / 2.0) * np.power(rho, l) * L
    return R


def spherical_harmonic_Ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray):
    """Complex spherical harmonic Y_{l,m}(theta,phi); orthonormal on S^2."""
    if not (l >= 0 and -l <= m <= l):
        raise ValueError("Quantum numbers (l,m) must satisfy l >= 0 and -l <= m <= l")

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Y = sp.sph_harm_y(l, m, theta, phi)
    return Y


def compute_psi_xz_slice(
    n: int,
    l: int,
    m: int,
    Z: int = 1,
    use_reduced_mass: bool = True,
    M: Optional[float] = None,
    extent_a_mu: float = 20.0,
    grid_points: int = 600,
    phi_value: float = 0.0,
    phi_mode: Literal["plane", "constant"] = "plane",
):
    """Evaluate psi_{n,l,m}(x,0,z) on the y=0 plane."""
    if not (n >= 1 and 0 <= l <= n - 1 and -l <= m <= l):
        raise ValueError("Quantum numbers (n,l,m) must satisfy n >= 1, 0 <= l <= n-1, and -l <= m <= l")

    mu = reduced_electron_nucleus_mass(Z, M) if use_reduced_mass else m_e
    a_mu = reduced_bohr_radius(mu)

    r_max = extent_a_mu * a_mu
    axis = np.linspace(-r_max, r_max, grid_points)
    Zg, Xg = np.meshgrid(axis, axis, indexing="ij")

    r = np.hypot(Xg, Zg)
    cos_theta = np.empty_like(r)
    np.divide(Zg, r, out=cos_theta, where=(r > 0))
    cos_theta[r == 0] = 1.0

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if phi_mode == "plane":
        phi = np.where(Xg >= 0.0, 0.0, np.pi)
    else:
        phi = np.full_like(r, float(phi_value))

    R = radial_wavefunction_Rnl(n, l, r, Z=Z, use_reduced_mass=use_reduced_mass, M=M)
    Y = spherical_harmonic_Ylm(l, m, theta, phi)
    psi = R * Y

    return Xg, Zg, psi, a_mu


def reduced_electron_nucleus_mass(Z: int, M: Optional[float] = None):
    """Compute electron-nucleus reduced mass mu."""
    if M is None:
        if Z == 1:
            M = m_p
        else:
            raise ValueError("'M' must be provided if Z > 1")

    return (m_e * M) / (m_e + M)


def reduced_bohr_radius(mu: float):
    """Compute Bohr radius with reduced mass."""
    a0 = physical_constants["Bohr radius"][0]
    return a0 * (m_e / mu)
