r"""exo50: a 50-parameter emission spectrum of a brown dwarf, in the style of petitRADTRANS retrievals.

θ → temperature profile, chemistry, clouds → emission spectrum → binned at R ≈ 100 over 0.85–4.95 μm (176 bins).
Same grids, bands and radiative transfer as exo30. The parameterisation follows pRT v4 retrievals of brown
dwarfs such as WISE 1049B:

- Temperature (Zhang et al. 2023): T_bottom at 100 bar and the gradient d ln T / d ln P at 10 nodes, equally
  spaced in log P from 100 bar to 1e-5 bar, interpolated linearly in log P and integrated upwards.
- Chemistry: H2O, CH4 and CO from [M/H] and C/O (equilibrium-like, as in exo6), quenched above P_quench (the
  abundances there are those at P_quench); 13CO from the isotope ratio 12C/13C; 20 trace species with free,
  vertically constant abundances.
- Clouds (Ackerman & Marley 2001, as in pRT's EddySed): MgSiO3(s) and Fe(s), each with a mass fraction X_cb at
  its base P_base decaying upwards as (P / P_base)^f_sed, and particles of radius r with a log-normal spread
  sigma_lnorm, shared by both. Patchy: a fraction f_cloud of the disc is cloudy, the rest clear (two radiative
  transfers per simulation).
- Haze: optical depth per dex at all pressures, ∝ λ^-alpha_haze.
- Instrument: the flux of the red detector (λ > 2.9 μm) is scaled by s_red, a nuisance parameter.
- Noise: Gaussian, fixed σ per bin (S/N ≈ 30 at the peak of the benchmark). pRT datasets often use relative
  noise, x (1 + ε / SNR); here it is additive, as in exo6 and exo30.
"""

import numpy as np
import torch

from . import exo30
from .exo30 import EDGES, P, WAVELENGTH, WL, deck, gas
from .physics import Constants, bands, binning_matrix, emission, flux_at_earth

N_NODES = 10
FREE = ["CO2", "NH3", "H2S", "HCN", "C2H2", "PH3", "SiO", "FeH", "CrH", "TiO", "VO", "K", "Na", "Cs", "OH", "HF",
        "HCl", "AlO", "CaH", "OCS"]
CLOUDS = ["MgSiO3", "Fe"]

# name, prior lower, prior upper, benchmark θ*, unit, description. The prior is uniform on [lower, upper].
PARAMETERS = (
    [("R_p", 0.8, 1.6, 1.0, "R_Jup", "planet radius"),
     ("log_g", 3.0, 5.5, 4.5, "log10 cm/s²", "surface gravity"),
     ("T_bottom", 2000.0, 8000.0, 4500.0, "K", "temperature at 100 bar")]
    + [(f"PTslope_{i}", 0.0, 0.45, s, "", f"d ln T / d ln P at node {i} (node 1: 100 bar, node 10: 1e-5 bar)")
       for i, s in enumerate([0.30, 0.28, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05], start=1)]
    + [("MH", -1.0, 1.0, 0.0, "dex", "metallicity [M/H]: C and O scale with 10^[M/H]"),
       ("CO", 0.2, 1.2, 0.55, "", "carbon-to-oxygen ratio"),
       ("log_Pquench", -3.0, 2.0, 0.5, "log10 bar", "quench pressure: H2O, CH4, CO frozen above it"),
       ("log_12C_13C", 1.0, 2.5, 1.95, "log10", "isotope ratio 12C / 13C, sets 13CO from CO")]
    + [(f"log_{s}", -12.0, -2.0, x, "log10 VMR", f"{s} volume mixing ratio, constant with altitude")
       for s, x in zip(FREE, [-5.5, -5.0, -4.5, -6.0, -8.0, -6.5, -6.0, -7.0, -8.5, -10.0, -10.0, -6.0, -5.5,
                              -9.0, -6.5, -7.0, -7.5, -10.0, -9.5, -8.0])]
    + [p for cloud, (x, pb, fs, r) in zip(CLOUDS, [(-3.0, 1.3, 2.0, 0.0), (-3.5, 1.6, 2.0, -0.3)]) for p in [
        (f"log_X_cb_{cloud}", -6.0, -1.0, x, "log10", f"{cloud}(s) mass fraction at the cloud base"),
        (f"log_Pbase_{cloud}", -1.0, 2.0, pb, "log10 bar", f"{cloud}(s) cloud-base pressure"),
        (f"f_sed_{cloud}", 0.5, 8.0, fs, "", f"{cloud}(s) sedimentation efficiency: X ∝ (P / P_base)^f_sed"),
        (f"log_cloud_radius_{cloud}", -2.0, 1.0, r, "log10 μm", f"{cloud}(s) median particle radius")]]
    + [("sigma_lnorm", 1.05, 3.0, 2.0, "", "width of the log-normal particle size distribution, both clouds"),
       ("f_cloud", 0.0, 1.0, 0.7, "", "fraction of the disc covered by clouds"),
       ("log_tau_haze", -4.0, 0.0, -2.5, "log10", "haze optical depth per dex of pressure, at 1 μm"),
       ("alpha_haze", 0.0, 6.0, 4.0, "", "haze slope: τ ∝ (λ / 1 μm)^-alpha_haze"),
       ("s_red", 0.9, 1.1, 1.02, "", "flux scale of the red detector (λ > 2.9 μm) relative to the blue one")]
)
SIGMA = 0.30025862877331094  # noise std per bin, in units of 1e-16 W m^-2 μm^-1

C_SUN, O_SUN = 5.4e-4, 9.8e-4  # number fractions relative to H2
K_CLOUD = 1000.0  # cloud opacity scale: both clouds of the benchmark are visible, neither hides the gas

_EQ = {  # equilibrium species and 13CO
    "H2O": exo30.BANDS["H2O"], "CH4": exo30.BANDS["CH4"], "CO": exo30.BANDS["CO"],
    "13CO": [(2.38, 0.04, 4.0), (4.78, 0.08, 30.0)],
}
_FREE_BANDS = dict(exo30.BANDS, **{
    "Na": [(0.82, 0.004, 0.8), (1.14, 0.003, 0.6)],
    "Cs": [(0.85, 0.003, 0.8), (0.89, 0.003, 0.6)],
    "OH": [(1.45, 0.04, 0.5), (2.9, 0.08, 3.0)],
    "HF": [(1.3, 0.02, 0.3), (2.5, 0.05, 2.0)],
    "HCl": [(1.75, 0.03, 0.3), (3.45, 0.07, 2.0)],
    "AlO": [(1.23, 0.04, 1.0), (1.65, 0.05, 0.4)],
    "CaH": [(0.9, 0.03, 1.0)],
    "OCS": [(2.05, 0.04, 0.5), (4.85, 0.06, 15.0)],
})
_nodes = np.linspace(2, -5, N_NODES)  # log10 bar, bottom to top
_const = Constants(
    P=P, dP=np.gradient(P), logP=np.log10(P), dlogP=np.gradient(P) / P / np.log(10), WL=WL,
    M=binning_matrix(WL, EDGES), CIA=exo30._const.arrays["CIA"],
    A=np.stack([np.interp(np.log10(P), _nodes[::-1], e[::-1]) for e in np.eye(N_NODES)]),  # slopes at layers
    SIGMA_EQ=np.stack([bands(WL, b, seed=200 + i) for i, b in enumerate(_EQ.values())]),
    SIGMA_X=np.stack([bands(WL, _FREE_BANDS[s], seed=100 + i) for i, s in enumerate(FREE)]),
    RED=(WAVELENGTH > 2.9).astype(float),
)


def profile(c, T_bottom: torch.Tensor, slopes: torch.Tensor) -> torch.Tensor:
    """Layer temperatures (B, L): ln T = ln T_bottom + ∫ slope d ln P, from the bottom up."""
    s = slopes @ c.A  # (B, L), layers top to bottom
    dlnP = torch.diff(torch.log(c.P))
    # integrate from the bottom layer upwards (trapezoid), then flip back
    steps = 0.5 * (s[:, 1:] + s[:, :-1]) * dlnP
    lnT = torch.log(T_bottom)[:, None] - torch.flip(torch.cumsum(torch.flip(steps, [1]), dim=1), [1])
    return torch.exp(torch.cat([lnT, torch.log(T_bottom)[:, None]], dim=1))  # bottom layer = 100 bar


def chemistry(c, T: torch.Tensor, MH, CO, log_Pq, log_iso) -> torch.Tensor:
    """Volume mixing ratios (B, L, 4) of H2O, CH4, CO, 13CO: equilibrium-like, quenched above P_quench."""
    C_tot = (C_SUN * 10 ** MH * CO / (C_SUN / O_SUN))[:, None]
    O_tot = (O_SUN * 10 ** MH)[:, None]
    f_ch4 = torch.sigmoid(-(T - (1100 + 120 * c.logP)) / 60)
    x_ch4 = C_tot * f_ch4
    x_co = torch.minimum(C_tot * (1 - f_ch4), O_tot)
    x_h2o = torch.clamp(O_tot - x_co, min=1e-12)
    x = torch.stack([x_h2o, x_ch4, x_co], dim=-1)
    near = torch.softmax(-((c.logP - log_Pq[:, None]) / 0.1) ** 2, dim=1)  # weights of the layers at P_quench
    x_q = torch.sum(near[..., None] * x, dim=1, keepdim=True)
    deep = torch.sigmoid((c.logP - log_Pq[:, None]) / 0.05)[..., None]
    x = deep * x + (1 - deep) * x_q
    x_13co = x[..., 2:3] / 10 ** log_iso[:, None, None]
    return torch.cat([x, x_13co], dim=-1)


def condensate(c, g, log_X, log_Pb, f_sed, log_r, sigma_lnorm) -> torch.Tensor:
    """Optical depth (B, L, λ) of one cloud: mass fraction X_cb (P / P_base)^f_sed above its base, particles of
    area-weighted radius r exp(2.5 ln² σ), extinction efficiency rising from Rayleigh (∝ x⁴) to 2 (x = 2πr / λ)."""
    above = torch.sigmoid(-(c.logP - log_Pb[:, None]) / 0.05)
    X = 10 ** log_X[:, None] * (c.P / 10 ** log_Pb[:, None]) ** f_sed[:, None] * above
    r = 10 ** log_r * torch.exp(2.5 * torch.log(sigma_lnorm) ** 2)
    x4 = (2 * np.pi * r[:, None] / c.WL) ** 4
    Q_over_r = 2 * x4 / (1 + x4) / r[:, None]  # (B, λ)
    return K_CLOUD * (X * c.dP * 1e6 / g[:, None])[..., None] * Q_over_r[:, None, :]


def flux(theta: torch.Tensor) -> torch.Tensor:
    """Noise-free binned spectra (B, 176) of parameters θ (B, 50)."""
    c = _const(theta)
    R_p, log_g, T_bottom = theta[:, 0], theta[:, 1], theta[:, 2]
    slopes = theta[:, 3:13]
    MH, CO, log_Pq, log_iso = theta[:, 13:17].unbind(-1)
    log_X = theta[:, 17:37]
    cl = theta[:, 37:45]
    sigma_lnorm, f_cloud, log_tau_h, alpha_h, s_red = theta[:, 45:50].unbind(-1)
    g = 10 ** log_g
    T = profile(c, T_bottom, slopes)
    # gas: equilibrium species (layer-dependent) + free species (constant) + collision-induced absorption
    x_eq = chemistry(c, T, MH, CO, log_Pq, log_iso)
    dtau_eq = exo30.MOL_SCALE * (x_eq @ c.SIGMA_EQ) * (c.dP * 1e6 / g[:, None])[..., None]
    dtau_clear = gas(c, log_g, log_X, c.SIGMA_X) + dtau_eq
    dtau_clear = dtau_clear + deck(c, torch.full_like(log_g, -10.0), log_tau_h, alpha_h)  # haze: all pressures
    dtau_cloudy = dtau_clear + sum(condensate(c, g, cl[:, 4 * k], cl[:, 4 * k + 1], cl[:, 4 * k + 2],
                                              cl[:, 4 * k + 3], sigma_lnorm) for k in range(len(CLOUDS)))
    f_c = f_cloud[:, None]
    I = f_c * emission(dtau_cloudy, T, c.WL) + (1 - f_c) * emission(dtau_clear, T, c.WL)
    F = flux_at_earth(I, R_p) @ c.M
    return F * (1 + c.RED * (s_red[:, None] - 1))
