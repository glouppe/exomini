from pathlib import Path

import numpy as np
import pytest
import torch

from toysbi import MODELS, Model


@pytest.fixture(params=MODELS, scope="module")
def model(request):
    return Model(request.param, device="cpu")


def test_parameters(model):
    assert len(model.names) == len(set(model.names)) == model.D == {"exo6": 6, "exo30": 30, "exo50": 50}[model.name]
    assert np.all(model.lower < model.upper)
    assert model.in_prior(model.theta_star)
    assert all(model.descriptions)
    assert all(f"`{n}`" in model.describe() for n in model.names)


def test_flux(model):
    theta = model.sample_prior(32, seed=0)
    f = model.flux(theta)
    assert f.shape == (32, model.n_bins)
    assert np.all(np.isfinite(f)) and np.all(f > 0)
    assert model.flux(theta[0]).shape == (1, model.n_bins)


def test_batches_and_precision(model):
    theta = model.sample_prior(10, seed=1)
    f = model.flux(theta, dtype=torch.float64)
    assert np.allclose(Model(model.name, device="cpu", batch_size=3).flux(theta, dtype=torch.float64), f)
    assert np.abs(model.flux(theta) - f).max() < 0.01 * model.sigma  # float32 is accurate enough


def test_noise(model):
    theta = np.repeat(model.theta_star[None], 500, axis=0)
    r = model.simulate(theta, seed=0) - model.flux(theta)
    assert abs(r.std() / model.sigma - 1) < 0.02


def test_observation(model):
    x = model.observation()
    assert np.array_equal(x, model.observation())
    chi2 = np.sum(((x - model.flux(model.theta_star)[0]) / model.sigma) ** 2)
    assert abs(chi2 - model.n_bins) < 5 * np.sqrt(2 * model.n_bins)


def test_likelihood(model):
    x = model.observation()
    theta = np.concatenate([model.theta_star[None], model.sample_prior(20, seed=2)])
    ll = model.log_likelihood(theta, x)
    r = (x - model.flux(theta)) / model.sigma
    gauss = -0.5 * np.sum(r ** 2 + np.log(2 * np.pi * model.sigma ** 2), axis=1)
    assert np.allclose(ll, gauss)
    assert ll[0] > ll[1:].max()  # the truth explains x_obs better than prior draws
    assert np.all(np.isinf(model.log_prior(model.upper + 1)))


def test_documented(model):
    """PARAMETERS.md lists every parameter with its prior (regenerate it when a PARAMETERS table changes)."""
    doc = (Path(__file__).parents[1] / "PARAMETERS.md").read_text()
    assert model.describe() in doc


def test_pairs(model):
    """The simulator still reproduces the shipped test pairs; pair 0 is the benchmark observation."""
    theta, x = model.test_pairs()
    f = np.load(Path(__file__).parents[1] / "toysbi" / "data" / f"{model.name}.npz")["f"]
    assert theta.shape == (8, model.D) and x.shape == (8, model.n_bins)
    assert np.allclose(theta[0], model.theta_star) and np.allclose(x[0], model.observation())
    assert np.abs(model.flux(theta, dtype=torch.float64) - f).max() < 1e-6 * model.sigma
