r"""Truncated sequential NPE (TSNPE, Deistler et al. 2022), then importance sampling.

    round 0      θ ~ prior                                     simulate, train q
    round r > 0  θ ~ prior restricted to where q(θ | x_obs)    simulate, retrain q on the pairs of all rounds
                 puts 1 - eps of its mass

The restricted prior is proportional to the prior inside the region, so plain maximum likelihood on the pooled
pairs still targets the posterior at x_obs (if the region holds all of it). q is then specific to x_obs.

    python tsnpe.py --model exo6
    python tsnpe.py --model exo30 --rounds 5 --sims 200000
"""

import json
import time

import numpy as np
import torch

from exomini import runs
from exomini.inference import NPE, fit, importance_sampling, truncated_prior


def main():
    p = runs.parser(__doc__, "tsnpe")
    p.add_argument("--rounds", type=int, default=4, help="number of rounds (default: 4)")
    p.add_argument("--sims", type=int, default=25_000, help="simulations per round (default: 25000)")
    p.add_argument("--eps", type=float, default=1e-4, help="mass of q left out of the region (default: 1e-4)")
    p.add_argument("--K", type=int, default=1024, help="candidates per draw of the truncated prior (default: 1024)")
    p.add_argument("--epochs", type=int, default=200, help="maximum training epochs per round (default: 200)")
    p.add_argument("--n-is", type=int, default=20_000, help="draws of q for importance sampling (default: 20000)")
    args = p.parse_args()
    model, x_obs, path = runs.setup(args)
    t0 = time.time()

    npe, thetas, fs, log = NPE(model), [], [], []
    theta = model.sample_prior(args.sims, seed=args.seed)
    for r in range(args.rounds):
        print(f"[round {r}] {args.sims} simulations, then training on {(r + 1) * args.sims}")
        thetas.append(theta)
        fs.append(model.flux(theta))
        # warm start from the previous round, smaller learning rate
        npe = fit(npe, np.concatenate(thetas), np.concatenate(fs), epochs=args.epochs,
                  lr=1e-3 if r == 0 else 5e-4, seed=args.seed + r)
        torch.save(npe.state_dict(), f"{path}/npe-{r}.pt")
        # How close is q(θ | x_obs) to the posterior? (diagnostic only: the next round does not use it)
        res = importance_sampling(npe, x_obs, args.n_is, seed=args.seed)
        info = {"round": r, "n_sims": (r + 1) * args.sims, "ess": res["ess"], "logz": res["logz"], **npe.history}
        print(f"  ESS {res['ess']:.0f} of {args.n_is}, log Z {res['logz']:.2f}")
        if r + 1 < args.rounds:
            t1 = time.time()
            theta, trunc = truncated_prior(npe, x_obs, args.sims, args.eps, args.K)
            info.update(trunc)
            print(f"  truncated prior: 10^{trunc['log10_volume']:.1f} of the prior volume, "
                  f"{trunc['ess_per_draw']:.1f} effective candidates per draw, {time.time() - t1:.0f} s")
        log.append(info)
        json.dump(log, open(f"{path}/rounds.json", "w"), indent=1)

    runs.save(path, res["samples"], res["q"], method=f"TSNPE, {args.rounds} rounds", n_sims=args.rounds * args.sims,
              ess=res["ess"], logz=res["logz"], q_inside_prior=res["q_inside_prior"], seconds=time.time() - t0)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
