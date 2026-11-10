r"""Compare the posteriors of several runs: corner plot, posterior predictive check, table.

    python compare.py --model exo6                             # every run in runs/exo6/
    python compare.py --model exo30 --runs nested tsnpe npe    # a selection, in this order
    python compare.py --model exo30 --only-is                  # NPE / TSNPE: reweighted samples only

Nested-sampling runs are filled grey (the first one is the reference of the table). For NPE and TSNPE, the
samples of q(θ | x_obs) are solid lines and their importance-sampled version dashed; the latter is left out when
its ESS is below 50, since it then rests on a handful of points.

Writes corner.png, ppc.png and table.md in runs/<model>/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.ndimage import gaussian_filter

from exomini import Model, runs

COLORS = ["#d1495b", "#1f5fae", "#e07b39", "#2a9d8f", "#8e5ea2", "#3a9a5b", "#b07d2b"]


def load_sets(folder: str, names: list, only_is: bool, min_ess: float = 50) -> list[dict]:
    """One dict per posterior to draw: label, samples, style, summary."""
    sets, k = [], 0
    for name in names:
        run = runs.load(os.path.join(folder, name))
        s = run["summary"]
        if run["q"] is None:  # nested sampling
            n_ref = sum(st["filled"] for st in sets)
            sets.append(dict(label=name, samples=run["samples"], color=["0.55", "0.25"][n_ref % 2],
                             filled=n_ref == 0, ls="-", summary=s, run=run))
            continue
        color, k = COLORS[k % len(COLORS)], k + 1
        if not (only_is and s["ess"] >= min_ess):
            sets.append(dict(label=f"{name}: q", samples=run["q"], color=color, filled=False, ls="-", summary=s,
                             run=run))
        if s["ess"] >= min_ess:
            sets.append(dict(label=f"{name}: q + IS", samples=run["samples"], color=color, filled=False, ls="--",
                             summary=s, run=run))
        else:
            print(f"{name}: importance sampling left out (ESS {s['ess']:.1f})")
    return sets


def levels(h: np.ndarray, masses=(0.95, 0.68)) -> list:
    """Density thresholds that enclose the given masses."""
    v = np.sort(h.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return [v[min(np.searchsorted(c, m), len(v) - 1)] for m in masses]


def corner(model: Model, sets: list, idx: list, truth, path: str, bins: int = 40):
    n = len(idx)
    size = 1.6 if n <= 12 else 0.9
    lo, up = model.lower, model.upper
    fig, axes = plt.subplots(n, n, figsize=(size * n + 1, size * n + 1), facecolor="white", squeeze=False)
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            ax = axes[a, b]
            if b > a:
                ax.axis("off")
                continue
            if a == b:  # 1D marginal
                edges = np.linspace(lo[i], up[i], bins + 1)
                centres = 0.5 * (edges[1:] + edges[:-1])
                for s in sets:
                    h = gaussian_filter(np.histogram(s["samples"][:, i], edges, density=True)[0], 1.0)
                    if s["filled"]:
                        ax.fill_between(centres, h, color=s["color"], alpha=0.5, lw=0)
                    else:
                        ax.plot(centres, h, color=s["color"], ls=s["ls"], lw=1.1)
                if truth is not None:
                    ax.axvline(truth[i], color="k", lw=0.8)
                ax.set_yticks([])
                ax.set_xlim(lo[i], up[i])
            else:  # 2D marginal, 68% and 95% contours
                for s in sets:
                    h, xe, ye = np.histogram2d(s["samples"][:, j], s["samples"][:, i], bins,
                                               [[lo[j], up[j]], [lo[i], up[i]]])
                    h = gaussian_filter(h, 1.2)
                    lv = levels(h)
                    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                    if s["filled"]:
                        ax.contourf(xc, yc, h.T, [lv[0], lv[1], h.max() + 1], colors=[s["color"]], alpha=0.35)
                        ax.contourf(xc, yc, h.T, [lv[1], h.max() + 1], colors=[s["color"]], alpha=0.45)
                    else:
                        ax.contour(xc, yc, h.T, lv, colors=s["color"], linestyles=s["ls"], linewidths=0.9)
                if truth is not None:
                    ax.plot(truth[j], truth[i], "k+", ms=6)
                ax.set_xlim(lo[j], up[j])
                ax.set_ylim(lo[i], up[i])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, prune="both"))
            if a != b:
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, prune="both"))
            fs = 9 if n <= 15 else 7
            if a == n - 1:
                ax.set_xlabel(model.names[j], fontsize=fs)
                ax.tick_params(axis="x", labelsize=fs - 2, rotation=45)
            else:
                ax.set_xticklabels([])
            if b == 0 and a > 0:
                ax.set_ylabel(model.names[i], fontsize=fs)
                ax.tick_params(axis="y", labelsize=fs - 2)
            else:
                ax.set_yticklabels([])
    handles = [matplotlib.patches.Patch(color=s["color"], alpha=0.5, label=s["label"]) if s["filled"] else
               matplotlib.lines.Line2D([], [], color=s["color"], ls=s["ls"], label=s["label"]) for s in sets]
    if truth is not None:
        handles.append(matplotlib.lines.Line2D([], [], color="k", marker="+", lw=0.8, label="truth"))
    fig.legend(handles=handles, loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"{model.name}: posteriors (contours: 68% and 95%)", fontsize=12, y=0.995)
    fig.subplots_adjust(hspace=0.06, wspace=0.06, left=0.07, bottom=0.07, right=0.99, top=0.96)
    fig.savefig(path, dpi=80 if n > 15 else 100)
    plt.close(fig)
    print(path)


def ppc(model: Model, sets: list, x_obs: np.ndarray, truth, path: str, n: int = 300) -> list:
    """Spectra of posterior draws against x_obs, and the χ² of x_obs under each draw. Returns the χ²."""
    rng = np.random.default_rng(0)
    wl, sigma = model.wavelength, model.sigma
    fig, (ax, axr, axc) = plt.subplots(3, 1, figsize=(11, 10), facecolor="white",
                                       gridspec_kw=dict(height_ratios=[2.2, 1, 1.4], hspace=0.35))
    ax.errorbar(wl, x_obs, sigma, fmt=".", ms=3, color="k", elinewidth=0.6, label="x_obs (±1σ)", zorder=5)
    chi2 = []
    for s in sets:
        f = model.flux(s["samples"][rng.choice(len(s["samples"]), min(n, len(s["samples"])), replace=False)])
        x_rep = f + sigma * rng.standard_normal(f.shape)  # replicated data: model + new noise
        chi_obs = np.sum(((x_obs - f) / sigma) ** 2, axis=1)
        p_value = np.mean(np.sum(((x_rep - f) / sigma) ** 2, axis=1) >= chi_obs)
        chi2.append(chi_obs)
        ax.plot(wl, np.median(f, 0), color=s["color"], ls=s["ls"], lw=1.0, label=s["label"])
        if s["filled"]:
            ax.fill_between(wl, *np.quantile(x_rep, [0.025, 0.975], axis=0), color=s["color"], alpha=0.2, lw=0)
        axr.plot(wl, np.median((x_obs - f) / sigma, 0), color=s["color"], ls=s["ls"], lw=0.9)
        axc.hist(chi_obs, 50, histtype="stepfilled" if s["filled"] else "step", color=s["color"],
                 alpha=0.5 if s["filled"] else 1, ls=s["ls"],
                 label=f"{s['label']}: median {np.median(chi_obs):.0f}, p = {p_value:.2f}")
    ax.set_ylabel("flux [1e-16 W m$^{-2}$ μm$^{-1}$]")
    ax.set_title("median model of each posterior (band: 95% of replicated data under the reference)")
    ax.legend(fontsize=8)
    axr.axhline(0, color="k", lw=0.6)
    for y in (-2, 2):
        axr.axhline(y, color="k", lw=0.4, ls=":")
    axr.set_ylabel("median (x_obs − f) / σ")
    axr.set_xlabel("wavelength [μm]")
    axc.axvline(model.n_bins, color="k", lw=0.8, ls=":", label=f"number of bins ({model.n_bins})")
    if truth is not None:
        chi_true = np.sum(((x_obs - model.flux(truth)[0]) / sigma) ** 2)
        axc.axvline(chi_true, color="k", lw=1.0, label=f"at the truth ({chi_true:.0f})")
    axc.set_xlabel("χ² of x_obs against posterior draws (p: posterior predictive p-value)")
    axc.set_yticks([])
    axc.legend(fontsize=7.5)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return chi2


def table(sets: list, chi2: list, path: str):
    """Cost, importance-sampling ESS, log Z, fit, and the marginals against the reference (first nested run):
    mean shift = mean over parameters of |Δ mean| / σ_ref, width ratio = median over parameters of σ / σ_ref."""
    ref = next((s["samples"] for s in sets if s["filled"]), None)
    rows = ["| posterior | simulations | ESS | log Z | median χ² | mean shift (σ) | width ratio |",
            "|---|---|---|---|---|---|---|"]
    for s, c in zip(sets, chi2):
        su = s["summary"]
        ess = f"{su['ess']:,.0f}" if "ess" in su else "–"
        logz = f"{su['logz']:.2f}" + (f" ± {su['logz_err']:.2f}" if "logz_err" in su else "")
        if s["ls"] == "-" and not s["filled"]:
            ess = logz = "–"  # q itself: the ESS and log Z belong to its reweighted version
        dm = ds = "–"
        if ref is not None and not s["filled"]:
            sd = ref.std(0)
            dm = f"{np.mean(np.abs(s['samples'].mean(0) - ref.mean(0)) / sd):.2f}"
            ds = f"{np.median(s['samples'].std(0) / sd):.2f}"
        rows.append(f"| {s['label']} | {su['n_sims']:,} | {ess} | {logz} | {np.median(c):.0f} | {dm} | {ds} |")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def main():
    p = runs.parser(__doc__)
    p.add_argument("--runs", nargs="+", help="run names in runs/<model>/ (default: all, nested sampling first)")
    p.add_argument("--only-is", action="store_true", help="NPE / TSNPE: draw the reweighted samples only")
    p.add_argument("--params", nargs="+", help="parameters of the corner plot (default: all if at most 12, "
                                               "else the 12 the first posterior constrains most)")
    p.add_argument("--out", default="runs", help="the runs are in OUT/<model>/ (default: runs)")
    args = p.parse_args()
    model = Model(args.model)
    folder = os.path.join(args.out, args.model)
    names = args.runs or sorted((r for r in os.listdir(folder) if os.path.exists(f"{folder}/{r}/summary.json")),
                                key=lambda r: (not r.startswith("nested"), r))
    sets = load_sets(folder, names, args.only_is)
    x_obs, truth = sets[0]["run"]["x"], sets[0]["run"]["theta"]
    for s in sets[1:]:
        if not np.allclose(s["run"]["x"], x_obs):
            raise ValueError(f"{s['label']} conditions on another observation than {sets[0]['label']}")

    if args.params:
        idx = [model.names.index(n) for n in args.params]
    elif model.D <= 12:
        idx = list(range(model.D))
    else:  # posterior std / prior std, under the first posterior
        idx = sorted(np.argsort(sets[0]["samples"].std(0) / (model.upper - model.lower))[:12])
    corner(model, sets, idx, truth, f"{folder}/corner.png")
    if model.D > 12 and not args.params:
        corner(model, sets, list(range(model.D)), truth, f"{folder}/corner-all.png")
    chi2 = ppc(model, sets, x_obs, truth, f"{folder}/ppc.png")
    table(sets, chi2, f"{folder}/table.md")


if __name__ == "__main__":
    main()
