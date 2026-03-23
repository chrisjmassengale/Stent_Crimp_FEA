"""
material.py — Simplified superelastic Nitinol constitutive model.

Implements a piecewise uniaxial stress-strain curve:

  Loading path:
    0   → ε_sAM  : linear elastic  (E_A ≈ 50 GPa austenite)
    ε_sAM → ε_fAM : upper plateau   (σ_AM_s → σ_AM_f, forward transformation)
    ε_fAM → max   : linear elastic  (E_M ≈ 25 GPa martensite)

  Unloading path:
    max  → ε_sMА : linear elastic  (E_M)
    ε_sMA → ε_fMA: lower plateau   (σ_MA_s → σ_MA_f, reverse transformation)
    ε_fMA → 0    : linear elastic  (E_A)

All stresses in Pa, strains dimensionless.

Reference parameters (Nitinol, body-temperature):
  E_A   = 50  GPa  (austenite Young's modulus)
  E_M   = 25  GPa  (martensite Young's modulus)
  σ_AM_s =  500 MPa  (upper plateau start stress)
  σ_AM_f =  550 MPa  (upper plateau end stress)   — slight slope
  ε_sAM  =  0.01     (strain at upper plateau start)
  ε_fAM  =  0.08     (strain at upper plateau end / full transformation)
  σ_MA_s =  250 MPa  (lower plateau start stress on unloading)
  σ_MA_f =  150 MPa  (lower plateau end stress)
  ε_sMA  = computed from σ_MA_s / E_M + ε_fAM residual
  ε_fMA  = computed from σ_MA_f / E_A  (~0.003)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NitinolParams:
    E_A:     float = 50.0e9    # Pa — austenite elastic modulus
    E_M:     float = 25.0e9    # Pa — martensite elastic modulus
    sig_AM_s: float = 500.0e6  # Pa — upper plateau start (loading)
    sig_AM_f: float = 550.0e6  # Pa — upper plateau end
    eps_AM_s: float = 0.01     # start of forward transformation
    eps_AM_f: float = 0.08     # end of forward transformation
    sig_MA_s: float = 250.0e6  # Pa — lower plateau start (unloading)
    sig_MA_f: float = 150.0e6  # Pa — lower plateau end
    # Derived strain thresholds on unloading (computed in __post_init__)
    eps_MA_s: float = field(init=False)
    eps_MA_f: float = field(init=False)

    def __post_init__(self):
        # On unloading from full transformation, reverse plateau starts when
        # σ drops to sig_MA_s.  The elastic unloading from ε_fAM uses E_M.
        self.eps_MA_s = self.eps_AM_f - (self.sig_AM_f - self.sig_MA_s) / self.E_M
        # Reverse plateau ends when σ reaches sig_MA_f; from there E_A carries load.
        self.eps_MA_f = self.sig_MA_f / self.E_A


NITINOL = NitinolParams()


# ── scalar stress-strain evaluator ────────────────────────────────────────────

def stress_from_strain(eps: float, eps_max: float,
                       params: NitinolParams = NITINOL) -> float:
    """
    Compute uniaxial stress (Pa) given current strain and the strain history
    maximum (determines whether we are on loading or unloading branch).

    Parameters
    ----------
    eps     : current axial strain (positive = tension, negative = compression
              handled by sign flip internally)
    eps_max : maximum |strain| reached so far in this element (for branch tracking)

    Returns
    -------
    sigma : stress in Pa (same sign as eps)
    """
    sign = 1.0 if eps >= 0.0 else -1.0
    e = abs(eps)
    em = abs(eps_max)
    p = params

    # Determine branch: loading if current |eps| >= prior max, unloading otherwise
    loading = (e >= em - 1e-12)

    if loading:
        sigma = _loading_sigma(e, p)
    else:
        # Unloading from em
        sigma_at_max = _loading_sigma(em, p)
        sigma = _unloading_sigma(e, em, sigma_at_max, p)

    return sign * sigma


def tangent_modulus(eps: float, eps_max: float,
                    params: NitinolParams = NITINOL) -> float:
    """
    Return the tangent modulus dσ/dε (Pa) at the given state.
    Used for the linearised element stiffness in each Newton iteration.
    """
    e = abs(eps)
    em = abs(eps_max)
    p = params
    loading = (e >= em - 1e-12)

    if loading:
        return _loading_tangent(e, p)
    else:
        return _unloading_tangent(e, em, p)


# ── internal helpers ───────────────────────────────────────────────────────────

def _loading_sigma(e: float, p: NitinolParams) -> float:
    if e <= p.eps_AM_s:
        return p.E_A * e
    elif e <= p.eps_AM_f:
        t = (e - p.eps_AM_s) / (p.eps_AM_f - p.eps_AM_s)
        return p.sig_AM_s + t * (p.sig_AM_f - p.sig_AM_s)
    else:
        return p.sig_AM_f + p.E_M * (e - p.eps_AM_f)


def _loading_tangent(e: float, p: NitinolParams) -> float:
    if e <= p.eps_AM_s:
        return p.E_A
    elif e <= p.eps_AM_f:
        return (p.sig_AM_f - p.sig_AM_s) / (p.eps_AM_f - p.eps_AM_s)
    else:
        return p.E_M


def _unloading_sigma(e: float, em: float, sigma_m: float,
                     p: NitinolParams) -> float:
    """Unloading from (em, sigma_m)."""
    # Phase 1: elastic unloading at E_M from the peak
    eps_plateau_start = em - (sigma_m - p.sig_MA_s) / p.E_M
    eps_plateau_start = max(eps_plateau_start, 0.0)

    if e >= eps_plateau_start:
        # Still in elastic unloading
        return sigma_m - p.E_M * (em - e)

    # Phase 2: lower plateau
    if e >= p.eps_MA_f:
        t = (e - p.eps_MA_f) / max(eps_plateau_start - p.eps_MA_f, 1e-12)
        return p.sig_MA_f + t * (p.sig_MA_s - p.sig_MA_f)

    # Phase 3: elastic unloading to zero at E_A
    return p.E_A * e


def _unloading_tangent(e: float, em: float, p: NitinolParams) -> float:
    sigma_m = _loading_sigma(em, p)
    eps_plateau_start = em - (sigma_m - p.sig_MA_s) / p.E_M
    eps_plateau_start = max(eps_plateau_start, 0.0)

    if e >= eps_plateau_start:
        return p.E_M
    elif e >= p.eps_MA_f:
        eps_range = max(eps_plateau_start - p.eps_MA_f, 1e-12)
        return (p.sig_MA_s - p.sig_MA_f) / eps_range
    else:
        return p.E_A


# ── vectorised versions ────────────────────────────────────────────────────────

def stress_array(eps_arr: np.ndarray, eps_max_arr: np.ndarray,
                 params: NitinolParams = NITINOL) -> np.ndarray:
    """Vectorised stress computation for an array of elements."""
    return np.array([stress_from_strain(float(e), float(em), params)
                     for e, em in zip(eps_arr, eps_max_arr)])


def tangent_array(eps_arr: np.ndarray, eps_max_arr: np.ndarray,
                  params: NitinolParams = NITINOL) -> np.ndarray:
    """Vectorised tangent modulus for an array of elements."""
    return np.array([tangent_modulus(float(e), float(em), params)
                     for e, em in zip(eps_arr, eps_max_arr)])


# ── quick self-test / diagnostic ──────────────────────────────────────────────

def plot_stress_strain_curve(params: NitinolParams = NITINOL,
                             n_pts: int = 400) -> None:
    """Print a text summary and optionally plot the σ-ε curve."""
    strains_load = np.linspace(0, 0.10, n_pts // 2)
    strains_unload = np.linspace(0.10, 0.0, n_pts // 2)

    eps_max_load = strains_load  # loading = monotonic increase
    eps_max = 0.10

    sig_load = np.array([stress_from_strain(e, e, params) for e in strains_load])
    sig_unload = np.array([stress_from_strain(e, eps_max, params) for e in strains_unload])

    print("Nitinol superelastic model - key points:")
    print(f"  Upper plateau start : eps={params.eps_AM_s:.3f}, sig={params.sig_AM_s/1e6:.0f} MPa")
    print(f"  Upper plateau end   : eps={params.eps_AM_f:.3f}, sig={params.sig_AM_f/1e6:.0f} MPa")
    print(f"  Lower plateau start : eps={params.eps_MA_s:.4f}, sig={params.sig_MA_s/1e6:.0f} MPa")
    print(f"  Lower plateau end   : eps={params.eps_MA_f:.4f}, sig={params.sig_MA_f/1e6:.0f} MPa")
    print(f"  E_A={params.E_A/1e9:.0f} GPa, E_M={params.E_M/1e9:.0f} GPa")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(strains_load * 100, sig_load / 1e6, 'b-', label='Loading')
        ax.plot(strains_unload * 100, sig_unload / 1e6, 'r--', label='Unloading')
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('Nitinol Superelastic Stress-Strain Curve')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
