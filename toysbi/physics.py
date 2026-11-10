r"""Physics shared by the toy forward models: opacity bands, Planck function, radiative transfer, binning.

Everything is batched in torch: a batch of B parameter vectors becomes arrays of shape (B, layers, wavelengths),
on a GPU when there is one. Fluxes are in units of 1e-16 W m^-2 μm^-1 at Earth.
"""

from types import SimpleNamespace

import numpy as np
import torch

H, C, KB = 6.626e-34, 2.998e8, 1.381e-23  # SI
R_JUP = 7.1492e7  # m
DISTANCE = 40 * 3.0857e16  # 40 pc, in m
UNIT = 1e-16  # flux unit, W m^-2 μm^-1


def bands(wl: np.ndarray, centers_widths_strengths: list, seed: int) -> np.ndarray:
    """A schematic cross-section: Gaussian bands with pseudo-line structure."""
    rng = np.random.default_rng(seed)
    s = np.zeros_like(wl)
    for c, w, a in centers_widths_strengths:
        s += a * np.exp(-0.5 * ((wl - c) / w) ** 2)
    lines = 1 + 0.6 * np.sin(2 * np.pi * wl / 0.004 + rng.uniform(0, 2 * np.pi)) * np.sin(2 * np.pi * wl / 0.0113)
    return s * np.clip(lines, 0.2, None) + 1e-4


def bins(lo: float, hi: float, resolution: float = 100) -> np.ndarray:
    """Bin edges with constant resolution λ / Δλ."""
    return np.exp(np.arange(np.log(lo), np.log(hi), 1 / resolution))


def binning_matrix(wl: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """M such that F @ M is the mean of F over the native points of each bin."""
    idx = np.digitize(wl, edges) - 1
    M = np.zeros((len(wl), len(edges) - 1))
    m = (idx >= 0) & (idx < len(edges) - 1)
    M[np.where(m)[0], idx[m]] = 1
    return M / M.sum(0, keepdims=True)


def emission(dtau: torch.Tensor, T: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
    """Emergent intensity of a non-scattering atmosphere, along μ = 2/3.

    dtau (B, L, λ): vertical optical depth of each layer, from the top down; T (B, L) in K; wl (λ,) in μm.
    Each layer emits B(T) and is attenuated by the layers above it.
    """
    mu = 2 / 3
    tau = torch.cumsum(dtau, dim=1) / mu
    w = torch.exp(-(tau - dtau / mu)) - torch.exp(-tau)
    wl = wl * 1e-6
    planck = 2 * H * C ** 2 / wl ** 5 / torch.expm1(H * C / (wl * KB * T[:, :, None]))  # W m^-2 m^-1 sr^-1
    return torch.sum(w * planck, dim=1)


def flux_at_earth(intensity: torch.Tensor, R_p: torch.Tensor) -> torch.Tensor:
    """Disc-integrated flux of a planet of radius R_p (R_Jup) at 40 pc, in units of UNIT."""
    return np.pi * intensity * ((R_p * R_JUP / DISTANCE) ** 2 * 1e-6 / UNIT)[:, None]


class Constants:
    """Numpy arrays, handed out as torch tensors on the device and dtype of a given tensor (cached)."""

    def __init__(self, **arrays):
        self.arrays = arrays
        self.cache = {}

    def __call__(self, like: torch.Tensor) -> SimpleNamespace:
        key = (like.device, like.dtype)
        if key not in self.cache:
            self.cache[key] = SimpleNamespace(**{
                k: torch.as_tensor(v, dtype=like.dtype, device=like.device) for k, v in self.arrays.items()})
        return self.cache[key]
