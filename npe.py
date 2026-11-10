r"""Amortised NPE: simulate from the prior, train q(θ | x) once, then sample q(θ | x_obs) and reweight the
samples with the exact likelihood (importance sampling).

    python npe.py --model exo6
    python npe.py --model exo30 --sims 1000000
"""

import time

import torch

from toysbi import runs
from toysbi.inference import NPE, fit, importance_sampling


def main():
    p = runs.parser(__doc__, "npe")
    p.add_argument("--sims", type=int, default=100_000, help="simulation budget (default: 100000)")
    p.add_argument("--epochs", type=int, default=200, help="maximum training epochs (default: 200)")
    p.add_argument("--n-is", type=int, default=20_000, help="draws of q for importance sampling (default: 20000)")
    args = p.parse_args()
    model, x_obs, path = runs.setup(args)
    t0 = time.time()

    print(f"[1/3] {args.sims} simulations from the prior")
    theta = model.sample_prior(args.sims, seed=args.seed)
    f = model.flux(theta)
    print(f"[2/3] training q(θ | x), {time.time() - t0:.0f} s")
    npe = fit(NPE(model), theta, f, epochs=args.epochs, seed=args.seed)
    torch.save(npe.state_dict(), f"{path}/npe.pt")
    print("[3/3] q(θ | x_obs) and importance sampling")
    res = importance_sampling(npe, x_obs, args.n_is, seed=args.seed)
    print(f"  ESS {res['ess']:.0f} of {args.n_is}, log Z {res['logz']:.2f}")
    runs.save(path, res["samples"], res["q"], method="NPE", n_sims=args.sims, ess=res["ess"], logz=res["logz"],
              q_inside_prior=res["q_inside_prior"], seconds=time.time() - t0, **npe.history)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
