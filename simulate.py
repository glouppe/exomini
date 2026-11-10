r"""Use a forward model directly: describe its parameters, simulate a dataset, or make an observation.

    python simulate.py --model exo50 --describe                   # parameters and priors (Markdown)
    python simulate.py --model exo6 --n 10000 --out data.npz      # (θ, x) pairs from the prior
    python simulate.py --model exo6 --theta 1300 4 0.5 0.8 0 1.2 --out obs.npz   # one observation of θ

An observation file can be given to the inference scripts with --obs obs.npz.
"""

import numpy as np

from exomini import Model, runs


def main():
    p = runs.parser(__doc__)
    p.add_argument("--describe", action="store_true", help="print the parameters and their priors")
    p.add_argument("--n", type=int, default=1000, help="number of prior draws (default: 1000)")
    p.add_argument("--theta", type=float, nargs="+", help="one parameter vector, in the order of --describe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data.npz", help="output file (default: data.npz)")
    args = p.parse_args()
    model = Model(args.model)

    if args.describe:
        print(model.describe())
        return
    if args.theta:
        theta = np.array(args.theta)
        if len(theta) != model.D:
            p.error(f"{model.name} has {model.D} parameters, got {len(theta)}")
        if not model.in_prior(theta):
            print("warning: θ is outside the prior")
        np.savez(args.out, x=model.observation(theta, seed=args.seed), theta=theta)
    else:
        theta = model.sample_prior(args.n, seed=args.seed)
        np.savez(args.out, theta=theta, x=model.simulate(theta, seed=args.seed), wavelength=model.wavelength,
                 names=model.names)
    print(f"{model} -> {args.out}")


if __name__ == "__main__":
    main()
