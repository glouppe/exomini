r"""Use a forward model directly: describe its parameters, simulate a dataset, or make an observation.

    python simulate.py --model exo50 --describe                   # parameters and priors (Markdown)
    python simulate.py --model exo6 --n 10000 --out data.npz      # (θ, x) pairs from the prior
    python simulate.py --model exo6 --theta 1300 4 0.5 0.8 0 1.2 --out obs.npz   # one observation of θ
    python simulate.py --model exo6 --plot parameters.png         # what each parameter does to the spectrum

An observation file can be given to the inference scripts with --obs obs.npz.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exomini import Model, runs
from exomini.plotting import BLUES, INK, MUTED, STYLE


def plot_parameters(model: Model, path: str):
    """One panel per parameter: the benchmark spectrum as that parameter sweeps its prior, the others at θ*."""
    cols = 3 if model.D <= 9 else 5
    rows = -(-model.D // cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.2 * rows), sharex=True, squeeze=False)
        star = model.flux(model.theta_star)[0]
        for i, ax in enumerate(axes.flat):
            if i >= model.D:
                ax.axis("off")
                continue
            theta = np.repeat(model.theta_star[None], len(BLUES), axis=0)
            theta[:, i] = np.linspace(model.lower[i], model.upper[i], len(BLUES))
            for f, color in zip(model.flux(theta), BLUES):
                ax.plot(model.wavelength, f, color=color, lw=1.0)
            ax.plot(model.wavelength, star, color=INK, lw=0.6, ls=":")
            ax.set_title(model.names[i], loc="left", fontsize=10, pad=13)
            unit = f" {model.units[i]}" if model.units[i] else ""
            ax.text(0, 1.02, f"{model.lower[i]:g} → {model.upper[i]:g}{unit}", transform=ax.transAxes,
                    va="bottom", fontsize=7.5, color=MUTED)
        for ax in axes[-1]:
            ax.set_xlabel("Wavelength [μm]")
        for ax in axes[:, 0]:
            ax.set_ylabel("Flux")
        for ax in axes.flat:
            ax.set_ylim(bottom=0)
        fig.suptitle(f"Each {model.name} parameter swept over its prior, light to dark (dotted: θ*)",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
        fig.savefig(path, dpi=90 if model.D > 12 else 110)
        plt.close(fig)


def main():
    p = runs.parser(__doc__)
    p.add_argument("--describe", action="store_true", help="print the parameters and their priors")
    p.add_argument("--plot", metavar="PNG", help="draw what each parameter does to the spectrum")
    p.add_argument("--n", type=int, default=1000, help="number of prior draws (default: 1000)")
    p.add_argument("--theta", type=float, nargs="+", help="one parameter vector, in the order of --describe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data.npz", help="output file (default: data.npz)")
    args = p.parse_args()
    model = Model(args.model)

    if args.describe:
        print(model.describe())
        return
    if args.plot:
        plot_parameters(model, args.plot)
        print(f"{model} -> {args.plot}")
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
