r"""A forward model with its prior, noise model and benchmark observation.

    >>> from toysbi import Model
    >>> m = Model("exo6")
    >>> theta = m.sample_prior(1000, seed=0)  # (1000, 6), uniform on [m.lower, m.upper]
    >>> x = m.simulate(theta, seed=0)         # (1000, 94), noisy spectra
    >>> x_obs = m.observation()               # the benchmark: θ* with one fixed noise draw
    >>> theta, x = m.test_pairs()             # 8 fixed (θ, x) pairs; pair 0 is the benchmark

Parameters θ and spectra x are numpy arrays; simulations run in torch, in batches, on a GPU if there is one.
"""

import importlib
from pathlib import Path

import numpy as np
import torch

MODELS = ("exo6", "exo30", "exo50")


class Model:
    """The forward model `name` (one of MODELS).

    Attributes: names, units, descriptions, lower, upper (prior bounds), theta_star (benchmark parameters),
    wavelength (bin centres, μm), sigma (noise std per bin), D (number of parameters), n_bins.
    """

    def __init__(self, name: str = "exo6", device: str | None = None, batch_size: int = 128):
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")
        self.name = name
        self.module = importlib.import_module(f"toysbi.{name}")
        names, lower, upper, star, units, descriptions = zip(*self.module.PARAMETERS)
        self.names, self.units, self.descriptions = list(names), list(units), list(descriptions)
        self.lower, self.upper, self.theta_star = np.array(lower), np.array(upper), np.array(star)
        self.wavelength = self.module.WAVELENGTH
        self.sigma = self.module.SIGMA
        self.D, self.n_bins = len(names), len(self.wavelength)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def __repr__(self):
        return f"Model({self.name!r}: {self.D} parameters, {self.n_bins} bins, device={self.device!r})"

    # prior: uniform on the box [lower, upper]
    def sample_prior(self, n: int, seed=None) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * np.random.default_rng(seed).random((n, self.D))

    def in_prior(self, theta: np.ndarray) -> np.ndarray:
        return np.all((theta >= self.lower) & (theta <= self.upper), axis=-1)

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        return np.where(self.in_prior(theta), -np.sum(np.log(self.upper - self.lower)), -np.inf)

    # simulator
    @torch.no_grad()
    def flux(self, theta: np.ndarray, dtype=torch.float32) -> np.ndarray:
        """Noise-free spectra (n, n_bins) of θ (n, D)."""
        theta = torch.as_tensor(np.atleast_2d(theta), dtype=dtype, device=self.device)
        f = [self.module.flux(theta[i:i + self.batch_size]) for i in range(0, len(theta), self.batch_size)]
        return torch.cat(f).cpu().double().numpy()

    def simulate(self, theta: np.ndarray, seed=None) -> np.ndarray:
        """Noisy spectra x = f(θ) + σ ε, ε ~ N(0, I)."""
        f = self.flux(theta)
        return f + self.sigma * np.random.default_rng(seed).standard_normal(f.shape)

    def log_likelihood(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """log p(x | θ) of the Gaussian noise model, normalised (for absolute log-evidences)."""
        r = (x - self.flux(theta)) / self.sigma
        return -0.5 * np.sum(r ** 2, axis=-1) - 0.5 * self.n_bins * np.log(2 * np.pi * self.sigma ** 2)

    def observation(self, theta: np.ndarray | None = None, seed: int = 7) -> np.ndarray:
        """A noisy spectrum of θ (default θ*), reproducible across machines (float64 on CPU)."""
        theta = self.theta_star if theta is None else np.asarray(theta, dtype=float)
        f = self.module.flux(torch.as_tensor(theta[None], dtype=torch.float64))[0].numpy()
        return f + self.sigma * np.random.default_rng(seed).standard_normal(f.shape)

    def test_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """8 fixed pairs (θ, x), shipped with the package, for reproducible comparisons. Pair 0 is the benchmark
        (θ*, as observation()); pairs 1-7 are prior draws."""
        d = np.load(Path(__file__).parent / "data" / f"{self.name}.npz")
        return d["theta"], d["x"]

    def describe(self) -> str:
        """The parameters and their priors, as a Markdown table."""
        rows = [f"| | parameter | prior | θ* | unit | description |", "|---|---|---|---|---|---|"]
        for i, (n, lo, up, s, u, d) in enumerate(zip(self.names, self.lower, self.upper, self.theta_star,
                                                     self.units, self.descriptions)):
            rows.append(f"| {i} | `{n}` | U({lo:g}, {up:g}) | {s:g} | {u} | {d} |")
        return "\n".join(rows)
