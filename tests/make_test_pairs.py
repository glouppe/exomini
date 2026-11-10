"""Writes exomini/data/<model>.npz: 8 fixed (θ, f, x) pairs per model, for reproducible comparisons.

Pair 0 is the benchmark (θ*, noise seed 7, as Model.observation()); pairs 1-7 are prior draws (seed 1), with
noise seeds 8-14. f is the noise-free spectrum in float64, x = f + σ ε. Rerun only if a model changes on purpose.
"""

import numpy as np
import torch

from exomini import MODELS, Model

for name in MODELS:
    m = Model(name, device="cpu")
    theta = np.concatenate([m.theta_star[None], m.sample_prior(7, seed=1)])
    f = m.flux(theta, dtype=torch.float64)
    x = np.stack([f[i] + m.sigma * np.random.default_rng(7 + i).standard_normal(m.n_bins) for i in range(8)])
    np.savez(f"exomini/data/{name}.npz", theta=theta, f=f, x=x)
    print(name, theta.shape, x.shape)
