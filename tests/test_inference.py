import numpy as np
import pytest

from toysbi import Model
from toysbi.inference import NPE, fit, importance_sampling, truncated_prior


@pytest.fixture(scope="module")
def trained():
    model = Model("exo6", device="cpu")
    theta = model.sample_prior(2000, seed=0)
    npe = fit(NPE(model), theta, model.flux(theta), epochs=3, verbose=False)
    return model, npe, model.observation()


def test_unit_transform():
    npe = NPE(Model("exo30", device="cpu"))
    theta = npe.model.sample_prior(100, seed=0)
    u = npe.to_unit(theta)
    assert np.all(np.abs(u) <= 1)
    assert np.allclose(npe.from_unit(u), theta)


def test_fit(trained):
    model, npe, x = trained
    assert np.isfinite(npe.history["valid_loss"]) and npe.history["epochs"] == 3
    theta, log_q = npe.sample(x, 100)
    assert theta.shape == (100, model.D) and np.all(np.isfinite(log_q))


def test_importance_sampling(trained):
    model, npe, x = trained
    res = importance_sampling(npe, x, 500)
    assert 1 <= res["ess"] <= 500
    assert np.isfinite(res["logz"])
    assert res["samples"].shape == (500, model.D) and np.all(model.in_prior(res["samples"]))


def test_truncated_prior(trained):
    model, npe, x = trained
    theta, info = truncated_prior(npe, x, 300, eps=1e-2, K=16)
    assert theta.shape == (300, model.D) and np.all(model.in_prior(theta))
    assert info["log10_volume"] <= 0 and 1 <= info["ess_per_draw"] <= 16
