r"""exo6: a 6-parameter emission spectrum of a warm giant planet, with self-consistent physics.

θ → temperature profile and abundances → emission spectrum → binned at R ≈ 100 over 0.95–2.45 μm (94 bins).

- Temperature: grey Eddington profile, T⁴ = ¾ T_int⁴ (τ + ⅔), with τ = κ_IR P / g.
- Chemistry: C and O scale with 10^[M/H]; carbon is in CH4 at low T / high P and in CO otherwise (smooth
  transition along a CO/CH4 boundary T(P)); the oxygen not in CO forms H2O.
- Opacity: schematic bands of H2O, CH4 and CO at their real wavelengths, H2–H2 collision-induced absorption
  (∝ P), and an opaque grey cloud deck below P_c.
- Noise: Gaussian, σ = 0.25e-16 W m^-2 μm^-1 per bin (S/N ≈ 30 at the peaks of the benchmark).
"""

import numpy as np
import torch

from .physics import Constants, bands, binning_matrix, bins, emission, flux_at_earth

# name, prior lower, prior upper, benchmark θ*, unit, description. The prior is uniform on [lower, upper].
PARAMETERS = [
    ("T_int", 700.0, 1700.0, 1100.0, "K", "internal temperature, sets the P–T profile"),
    ("log_g", 3.0, 5.0, 3.75, "log10 cm/s²", "surface gravity"),
    ("MH", -1.0, 1.0, 0.0, "dex", "metallicity [M/H]: C and O scale with 10^[M/H]"),
    ("CO", 0.2, 1.2, 0.55, "", "carbon-to-oxygen ratio"),
    ("log_Pc", -1.5, 1.5, 0.5, "log10 bar", "pressure at the top of an opaque grey cloud deck"),
    ("R_p", 0.8, 1.6, 1.0, "R_Jup", "planet radius"),
]
SIGMA = 0.25  # noise std per bin, in units of 1e-16 W m^-2 μm^-1

P = np.logspace(-5, 2, 80)  # bar, top to bottom
WL = np.linspace(0.9, 2.5, 3000)  # μm, native grid
EDGES = bins(0.95, 2.45)
WAVELENGTH = np.sqrt(EDGES[:-1] * EDGES[1:])  # μm, bin centres

C_SUN, O_SUN = 5.4e-4, 9.8e-4  # number fractions relative to H2
KAPPA_IR = 3e-2  # grey opacity of the P–T profile, cm² / g
MOL_SCALE = 50.0  # strong bands reach τ = 1 near 0.01 bar
CIA_SCALE = 0.5  # windows reach τ = 1 near 1 bar

_const = Constants(
    P=P, dP=np.gradient(P), logP=np.log10(P), WL=WL, M=binning_matrix(WL, EDGES),
    H2O=bands(WL, [(0.94, 0.02, 0.3), (1.13, 0.03, 1.0), (1.39, 0.045, 6.0), (1.88, 0.06, 10.0),
                   (2.65, 0.09, 20.0)], seed=1),
    CH4=bands(WL, [(1.17, 0.02, 0.4), (1.38, 0.03, 1.0), (1.67, 0.035, 3.0), (2.20, 0.04, 2.0),
                   (2.32, 0.06, 8.0)], seed=2),
    CO=bands(WL, [(1.58, 0.02, 0.05), (2.33, 0.04, 4.0)], seed=3),
    CIA=0.02 * np.exp(-0.5 * ((WL - 2.15) / 0.25) ** 2) + 0.008 * np.exp(-0.5 * ((WL - 1.2) / 0.15) ** 2),
)


def flux(theta: torch.Tensor) -> torch.Tensor:
    """Noise-free binned spectra (B, 94) of parameters θ (B, 6)."""
    c = _const(theta)
    T_int, log_g, MH, CO, log_Pc, R_p = theta.unbind(-1)
    g = (10 ** log_g)[:, None]
    # temperature and chemistry, (B, L)
    T = (0.75 * T_int[:, None] ** 4 * (KAPPA_IR * c.P * 1e6 / g + 2 / 3)) ** 0.25
    C_tot = (C_SUN * 10 ** MH * CO / (C_SUN / O_SUN))[:, None]
    O_tot = (O_SUN * 10 ** MH)[:, None]
    f_ch4 = torch.sigmoid(-(T - (1100 + 120 * c.logP)) / 60)
    x_ch4 = C_tot * f_ch4
    x_co = torch.minimum(C_tot * (1 - f_ch4), O_tot)
    x_h2o = torch.clamp(O_tot - x_co, min=1e-12)
    # opacity, (B, L, λ)
    kappa = MOL_SCALE * (x_h2o[..., None] * c.H2O + x_ch4[..., None] * c.CH4 + x_co[..., None] * c.CO)
    kappa = kappa + CIA_SCALE * c.P[:, None] * c.CIA
    dtau = kappa * (c.dP[:, None] * 1e6 / g[..., None])
    cloud = 50.0 * torch.sigmoid((c.logP - log_Pc[:, None]) / 0.05)  # opaque below P_c, smooth over 0.1 dex
    dtau = dtau + (cloud * c.dP / c.P)[..., None]
    return flux_at_earth(emission(dtau, T, c.WL), R_p) @ c.M
