r"""Neural posterior estimation, truncated proposals and importance sampling.

The estimator works in u = 2 (θ - lower) / (upper - lower) - 1, where the prior is uniform on [-1, 1]^D.

    >>> from exomini import Model
    >>> from exomini.inference import NPE, fit, importance_sampling
    >>> m = Model("exo6")
    >>> theta = m.sample_prior(10_000, seed=0)
    >>> npe = fit(NPE(m), theta, m.flux(theta))                # fresh noise is drawn at every epoch
    >>> post = importance_sampling(npe, m.observation(), 10_000)  # samples of q, reweighted with the likelihood
    >>> post["ess"], post["logz"]
"""

import time

import numpy as np
import torch
import torch.nn as nn
import zuko

from .model import Model


class NPE(nn.Module):
    r"""q(θ | x): an MLP summarises the spectrum, a neural spline flow models u given the summary.

    Spectra span orders of magnitude, so the input is asinh(x / σ): linear within the noise, logarithmic above.
    The flow is a coupling flow (passes=2): sampling costs one pass per transform, against D for a fully
    autoregressive flow, which matters for truncated proposals.
    """

    def __init__(self, model: Model, embedding: int = 128, transforms: int = 5, hidden: tuple = (128, 128)):
        super().__init__()
        self.model = model
        self.embed = nn.Sequential(
            nn.Linear(model.n_bins, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, embedding))
        self.flow = zuko.flows.NSF(model.D, embedding, transforms=transforms, passes=2, hidden_features=hidden)

    def to_unit(self, theta: np.ndarray) -> np.ndarray:
        return 2 * (theta - self.model.lower) / (self.model.upper - self.model.lower) - 1

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.model.lower + (u + 1) / 2 * (self.model.upper - self.model.lower)

    def forward(self, x: torch.Tensor) -> zuko.distributions.Distribution:
        """The distribution q(u | x)."""
        return self.flow(self.embed(torch.asinh(x / self.model.sigma)))

    @torch.no_grad()
    def sample(self, x: np.ndarray, n: int, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        """n draws θ ~ q(θ | x) and their log-densities log q(θ | x), in chunks to bound memory."""
        q = self(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        u, log_q = zip(*(q.rsample_and_log_prob((min(chunk, n - i),)) for i in range(0, n, chunk)))
        u, log_q = torch.cat(u).double().cpu().numpy(), torch.cat(log_q).double().cpu().numpy()
        log_jacobian = np.sum(np.log(2 / (self.model.upper - self.model.lower)))  # log |du / dθ|
        return self.from_unit(u), log_q + log_jacobian

    @property
    def device(self):
        return next(self.parameters()).device


def fit(npe: NPE, theta: np.ndarray, f: np.ndarray, epochs: int = 200, patience: int = 15,
        batch_size: int = 512, lr: float = 1e-3, seed: int = 0, verbose: bool = True) -> NPE:
    r"""Train q(θ | x) by maximum likelihood on pairs (θ, f(θ)) of noise-free spectra, in place.

    Each epoch draws new noise, x = f + σ ε: the noise model is known, so this is exact, and it is a free data
    augmentation (each simulation is seen with many noise realisations). The last 5% of the pairs, with one fixed
    noise draw, are held out for early stopping; the best model on them is kept. Statistics in npe.history.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    npe.to(dev)
    sigma = npe.model.sigma
    u = torch.as_tensor(npe.to_unit(theta), dtype=torch.float32, device=dev)
    f = torch.as_tensor(f, dtype=torch.float32, device=dev)
    n_val = max(1, len(u) // 20)
    u_tr, f_tr, u_va = u[:-n_val], f[:-n_val], u[-n_val:]
    x_va = f[-n_val:] + sigma * torch.randn(f[-n_val:].shape, generator=torch.Generator(dev).manual_seed(1),
                                            device=dev)
    opt = torch.optim.AdamW(npe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 3)
    best, best_state, wait, t0 = np.inf, None, 0, time.time()
    for epoch in range(epochs):
        npe.train()
        losses = []
        for j in torch.randperm(len(u_tr), device=dev).split(batch_size):
            x = f_tr[j] + sigma * torch.randn_like(f_tr[j])
            loss = -npe(x).log_prob(u_tr[j]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(npe.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        npe.eval()
        with torch.no_grad():
            val = torch.cat([npe(x_va[j]).log_prob(u_va[j]) for j in torch.arange(n_val, device=dev).split(4096)])
            val = -val.mean().item()
        sched.step(val)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train {np.mean(losses):7.3f}  valid {val:7.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:.0f} s", flush=True)
        if val < best - 1e-3:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in npe.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    npe.load_state_dict(best_state)
    npe.history = {"valid_loss": best, "epochs": epoch + 1, "train_seconds": time.time() - t0}
    if verbose:
        print(f"  best valid {best:.3f} after {epoch + 1} epochs, {time.time() - t0:.0f} s", flush=True)
    return npe


def importance_sampling(npe: NPE, x: np.ndarray, n: int = 20_000, seed: int = 0) -> dict:
    r"""Reweight n draws of q(θ | x) by w = p(x | θ) p(θ) / q(θ | x).

    If q were the exact posterior every weight would be equal. The effective sample size, ESS = (Σw)² / Σw²,
    tells how far it is: ESS ≈ n means q is accurate; an ESS of a few means that neither the q samples nor their
    reweighted version can be trusted. The mean weight estimates the evidence Z = p(x).

    Returns a dict: q (draws of q inside the prior), samples (n draws resampled by weight), ess, logz,
    q_inside_prior (fraction of q's draws inside the prior box).
    """
    model = npe.model
    theta, log_q = npe.sample(x, n)
    inside = model.in_prior(theta)
    log_w = np.full(n, -np.inf)
    log_w[inside] = model.log_likelihood(theta[inside], x) + model.log_prior(theta[inside]) - log_q[inside]
    w = np.exp(log_w - log_w.max())
    ess = w.sum() ** 2 / (w ** 2).sum()
    logz = log_w.max() + np.log(w.mean())
    idx = np.random.default_rng(seed).choice(n, size=n, p=w / w.sum())
    return {"q": theta[inside], "samples": theta[idx], "ess": float(ess), "logz": float(logz),
            "q_inside_prior": float(inside.mean())}


@torch.no_grad()
def truncated_prior(npe: NPE, x: np.ndarray, n: int, eps: float = 1e-4, K: int = 1024) -> tuple[np.ndarray, dict]:
    r"""n draws from the prior truncated to the highest-probability region of q(θ | x) with mass 1 - eps.

    The region is {θ : q(θ | x) > τ}, τ the eps-quantile of log q under q. Draws are made by sampling-importance-
    resampling: K candidates from q per draw, weights 1[θ in the prior box and in the region] p(θ) / q(θ | x),
    one candidate kept in proportion to its weight. Exact as K → ∞ (Deistler et al. 2022 use K = 1024).

    Returns the draws and a dict: tau, log10_volume (fraction of the prior volume in the region), and
    ess_per_draw (effective number of candidates per draw: small values mean heavy-tailed weights).
    """
    model = npe.model
    _, lq = npe.sample(x, max(200_000, int(20 / eps)))
    tau = np.quantile(lq, eps)
    log_p = -np.sum(np.log(model.upper - model.lower))  # the uniform prior density
    out, ess, log_vol = [], [], []
    per = max(1, min(2_000_000 // K, n))  # draws per batch
    while sum(len(o) for o in out) < n:
        theta, lq = npe.sample(x, per * K)
        log_w = np.where(model.in_prior(theta) & (lq > tau), log_p - lq, -np.inf).reshape(per, K)
        log_vol.append(np.logaddexp.reduce(log_w.ravel()) - np.log(per * K))  # E_q[p 1[region] / q]
        ok = np.isfinite(log_w).any(axis=1)
        w = np.exp(log_w[ok] - log_w[ok].max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        ess.append(np.mean(1 / (w ** 2).sum(axis=1)))
        rng = np.random.default_rng(len(out))
        j = (w.cumsum(axis=1) > rng.random((len(w), 1))).argmax(axis=1)
        out.append(theta.reshape(per, K, -1)[ok][np.arange(len(j)), j])
    info = {"tau": float(tau), "log10_volume": float(np.mean(log_vol) / np.log(10)),
            "ess_per_draw": float(np.mean(ess))}
    return np.concatenate(out)[:n], info
