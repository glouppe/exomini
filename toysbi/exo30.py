r"""exo30: a 30-parameter emission spectrum, shaped like a free retrieval.

θ → temperature profile, constant abundances, cloud → emission spectrum → binned at R ≈ 100 over 0.85–4.95 μm
(176 bins, a NIRSpec PRISM-like range).

- Temperature: free profile with 10 nodes equally spaced in log P from 100 bar to 1e-5 bar. Parameterised by the
  bottom temperature T_0 and nine ratios T_i / T_{i-1}, linear in log P between nodes, floored at 100 K.
- Chemistry: 15 species with free, vertically constant abundances. Bands are schematic, at roughly the real
  positions, so that some overlap (HCN / C2H2 / NH3 at 3 μm, CO2 / PH3 at 4.3 μm) and some are barely visible:
  many parameters are prior-dominated, as in real retrievals.
- Cloud: a non-grey deck below P_c, with optical depth τ_c per dex of pressure and slope τ ∝ λ^-α.
- Noise: Gaussian, σ = 0.39e-16 W m^-2 μm^-1 per bin (S/N ≈ 30 at the peak of the benchmark).
"""

import numpy as np
import torch

from .physics import Constants, bands, binning_matrix, bins, emission, flux_at_earth

SPECIES = ["H2O", "CH4", "CO", "CO2", "NH3", "H2S", "HCN", "C2H2", "PH3", "FeH", "CrH", "TiO", "VO", "K", "SiO"]
N_NODES = 10

# name, prior lower, prior upper, benchmark θ*, unit, description. The prior is uniform on [lower, upper].
PARAMETERS = (
    [("log_g", 3.0, 5.0, 3.75, "log10 cm/s²", "surface gravity"),
     ("R_p", 0.8, 1.6, 1.0, "R_Jup", "planet radius"),
     ("T_0", 1500.0, 6000.0, 4900.0, "K", "temperature at the bottom node, 100 bar")]
    + [(f"r_{i}", 0.55, 1.05, r, "", f"T_{i} / T_{i - 1}, temperature ratio between P–T nodes {i - 1} and {i}")
       for i, r in enumerate([0.62, 0.66, 0.72, 0.80, 0.88, 0.95, 0.98, 1.0, 1.0], start=1)]
    + [(f"log_{s}", -10.0, -2.0, x, "log10 VMR", f"{s} volume mixing ratio, constant with altitude")
       for s, x in zip(SPECIES, [-3.3, -3.6, -3.5, -5.0, -4.5, -4.5, -5.0, -6.0, -5.5, -6.0, -6.5, -8.0, -8.0,
                                 -5.5, -5.0])]
    + [("log_Pc", -2.0, 2.0, 0.5, "log10 bar", "pressure at the top of the cloud deck"),
       ("log_tau_c", -1.0, 2.0, 0.5, "log10", "cloud optical depth per dex of pressure, at 1 μm"),
       ("alpha_c", 0.0, 4.0, 1.0, "", "cloud slope: τ ∝ (λ / 1 μm)^-alpha_c")]
)
SIGMA = 0.3916051966046202  # noise std per bin, in units of 1e-16 W m^-2 μm^-1

# 40 layers and 4000 wavelengths (4 points per pseudo-line period): within 0.7σ of a 80 x 8000 grid.
P = np.logspace(-5, 2, 40)
WL = np.linspace(0.8, 5.0, 4000)
EDGES = bins(0.85, 4.95)
WAVELENGTH = np.sqrt(EDGES[:-1] * EDGES[1:])

# (centre μm, width μm, strength), strengths relative to H2O's 1.88 μm band = 10
BANDS = {
    "H2O": [(0.94, 0.02, 0.3), (1.13, 0.03, 1.0), (1.39, 0.045, 6.0), (1.88, 0.06, 10.0), (2.7, 0.09, 20.0),
            (3.2, 0.1, 2.0)],
    "CH4": [(1.17, 0.02, 0.4), (1.38, 0.03, 1.0), (1.67, 0.035, 3.0), (2.20, 0.04, 2.0), (2.32, 0.06, 8.0),
            (3.31, 0.08, 30.0)],
    "CO": [(1.58, 0.02, 0.05), (2.33, 0.04, 4.0), (4.67, 0.08, 30.0)],
    "CO2": [(1.43, 0.015, 0.05), (1.6, 0.02, 0.1), (2.0, 0.03, 0.5), (2.7, 0.04, 3.0), (4.27, 0.05, 80.0)],
    "NH3": [(1.5, 0.04, 0.5), (2.0, 0.04, 0.8), (3.0, 0.08, 4.0)],
    "H2S": [(1.59, 0.03, 0.1), (2.6, 0.05, 1.0), (3.7, 0.08, 0.5)],
    "HCN": [(1.53, 0.02, 0.3), (3.0, 0.04, 10.0)],
    "C2H2": [(1.52, 0.02, 0.1), (3.0, 0.05, 6.0)],
    "PH3": [(2.15, 0.04, 0.5), (4.3, 0.08, 10.0)],
    "FeH": [(0.99, 0.01, 3.0), (1.2, 0.03, 1.0), (1.6, 0.03, 0.5)],
    "CrH": [(0.87, 0.02, 2.0), (1.0, 0.02, 1.0)],
    "TiO": [(0.85, 0.05, 5.0), (0.95, 0.05, 3.0), (1.1, 0.06, 1.0)],
    "VO": [(1.05, 0.04, 3.0), (1.2, 0.04, 1.0)],
    "K": [(1.17, 0.003, 0.8), (1.25, 0.003, 1.0)],
    "SiO": [(4.05, 0.1, 5.0)],
}
MOL_SCALE = 50.0
CIA_SCALE = 0.5


def interpolation(n_nodes: int) -> np.ndarray:
    """A (nodes, layers): node temperatures @ A = layer temperatures, linear in log P between the nodes."""
    nodes = np.linspace(2, -5, n_nodes)  # log10 bar, bottom to top
    return np.stack([np.interp(np.log10(P), nodes[::-1], e[::-1]) for e in np.eye(n_nodes)])


_const = Constants(
    P=P, dP=np.gradient(P), logP=np.log10(P), dlogP=np.gradient(P) / P / np.log(10), WL=WL,
    M=binning_matrix(WL, EDGES), A=interpolation(N_NODES),
    SIGMA_X=np.stack([bands(WL, BANDS[s], seed=i + 1) for i, s in enumerate(SPECIES)]),
    CIA=0.02 * np.exp(-0.5 * ((WL - 2.15) / 0.25) ** 2) + 0.008 * np.exp(-0.5 * ((WL - 1.2) / 0.15) ** 2)
    + 0.01 * np.exp(-0.5 * ((WL - 4.5) / 0.6) ** 2),
)


def profile(T_0: torch.Tensor, ratios: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Layer temperatures (B, L) from the bottom temperature and the node-to-node ratios."""
    T_nodes = T_0[:, None] * torch.cat([torch.ones_like(T_0[:, None]), torch.cumprod(ratios, dim=-1)], dim=-1)
    return torch.clamp(T_nodes @ A, min=100.0)


def gas(c, log_g: torch.Tensor, log_X: torch.Tensor, sigma_x: torch.Tensor) -> torch.Tensor:
    """Optical depth (B, L, λ) of the molecules (constant abundances) and of collision-induced absorption."""
    kappa = ((10 ** log_X) @ sigma_x * MOL_SCALE)[:, None, :] + CIA_SCALE * c.P[:, None] * c.CIA
    return kappa * c.dP[:, None] * 1e6 / (10 ** log_g)[:, None, None]


def deck(c, log_P_top: torch.Tensor, log_tau: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Optical depth (B, L, λ) of a cloud: 10^log_tau per dex below P_top, ∝ λ^-alpha."""
    below = torch.sigmoid((c.logP - log_P_top[:, None]) / 0.05)
    return ((10 ** log_tau)[:, None] * below * c.dlogP)[..., None] * c.WL ** (-alpha[:, None, None])


def flux(theta: torch.Tensor) -> torch.Tensor:
    """Noise-free binned spectra (B, 176) of parameters θ (B, 30)."""
    c = _const(theta)
    n, s = N_NODES - 1, len(SPECIES)
    log_g, R_p, T_0 = theta[:, 0], theta[:, 1], theta[:, 2]
    ratios, log_X = theta[:, 3:3 + n], theta[:, 3 + n:3 + n + s]
    log_Pc, log_tau_c, alpha_c = theta[:, -3:].unbind(-1)
    T = profile(T_0, ratios, c.A)
    dtau = gas(c, log_g, log_X, c.SIGMA_X) + deck(c, log_Pc, log_tau_c, alpha_c)
    return flux_at_earth(emission(dtau, T, c.WL), R_p) @ c.M
