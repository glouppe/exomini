r"""Compare the posteriors of several runs: corner plot, posterior predictive check, table.

    python compare.py --model exo6                             # every run in runs/exo6/
    python compare.py --model exo30 --runs nested tsnpe npe    # a selection, in this order
    python compare.py --model exo30 --only-is                  # NPE / TSNPE: reweighted samples only

Each run has its own colour; no run is treated as the truth. For NPE and TSNPE, the samples of q(θ | x_obs) are
solid lines and their importance-sampled version dashed; the latter is left out when its ESS is below 50, since it
then rests on a handful of points. The table compares the marginals with one run (--against, default: nested).

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
from exomini.plotting import COLORS, FAINT, INK, MUTED, STYLE


LABELS = {"npe": "NPE", "tsnpe": "TSNPE", "nested": "Nested sampling"}  # run name -> display name


def load_sets(folder: str, names: list, only_is: bool, min_ess: float = 50) -> list[dict]:
    """One dict per posterior to draw: label, samples, style, summary."""
    sets = []
    for k, name in enumerate(names):
        run = runs.load(os.path.join(folder, name))
        s, color = run["summary"], COLORS[k % len(COLORS)]
        label = LABELS.get(name, name)
        common = dict(name=name, color=color, summary=s, run=run)
        if run["q"] is None:  # nested sampling
            sets.append(dict(label=label, samples=run["samples"], ls="-", q=False, **common))
            continue
        if not (only_is and s["ess"] >= min_ess):
            sets.append(dict(label=f"{label}, q", samples=run["q"], ls="-", q=True, **common))
        if s["ess"] >= min_ess:
            sets.append(dict(label=f"{label}, q + IS", samples=run["samples"], ls="--", q=False, **common))
        else:
            print(f"{name}: importance sampling left out (ESS {s['ess']:.1f})")
    return sets


def levels(h: np.ndarray, masses=(0.95, 0.68)) -> list:
    """Density thresholds that enclose the given masses."""
    v = np.sort(h.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return [v[min(np.searchsorted(c, m), len(v) - 1)] for m in masses]


def view(model: Model, sets: list, idx: list) -> tuple[np.ndarray, np.ndarray]:
    """Axis ranges: where the posteriors are (0.1-99.9% quantiles, padded), within the prior."""
    lo, up = np.quantile(np.concatenate([s["samples"][:, idx] for s in sets]), [0.001, 0.999], axis=0)
    pad = 0.2 * (up - lo) + 1e-9
    return np.maximum(lo - pad, model.lower[idx]), np.minimum(up + pad, model.upper[idx])


def axis_label(model: Model, i: int, units: bool) -> str:
    return f"{model.names[i]} [{model.units[i]}]" if units and model.units[i] else model.names[i]


def corner(model: Model, sets: list, idx: list, truth, path: str, spectrum=None, bins: int = 50):
    """Pairwise marginals of the parameters idx; spectrum = (x_obs, predictions), drawn in the empty corner."""
    n = len(idx)
    size = 1.9 if n <= 8 else 1.6 if n <= 12 else 0.9
    fs = 9 if n <= 15 else 7
    lo, up = view(model, sets, idx)
    left, bottom, right, top = 0.09, 0.08, 0.98, 0.97
    fig, axes = plt.subplots(n, n, figsize=(size * n + 0.8, size * n + 0.8), squeeze=False)
    fig.subplots_adjust(hspace=0.08, wspace=0.08, left=left, bottom=bottom, right=right, top=top)
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            ax = axes[a, b]
            if b > a:
                ax.axis("off")
                continue
            if a == b:  # 1D marginal
                edges = np.linspace(lo[a], up[a], bins + 1)
                centres = 0.5 * (edges[1:] + edges[:-1])
                for s in sets:
                    h = gaussian_filter(np.histogram(s["samples"][:, i], edges, density=True)[0], 1.0)
                    ax.plot(centres, h, color=s["color"], ls=s["ls"], lw=1.3)
                if truth is not None:
                    ax.axvline(truth[i], color=INK, lw=0.8)
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            else:  # 2D marginal, 68% and 95% contours
                rng = [[lo[b], up[b]], [lo[a], up[a]]]
                for s in sets:
                    h, xe, ye = np.histogram2d(s["samples"][:, j], s["samples"][:, i], bins, rng)
                    h = gaussian_filter(h, 1.3)
                    lv = levels(h)
                    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                    ax.contour(xc, yc, h.T, lv, colors=s["color"], linestyles=s["ls"], linewidths=1.1)
                if truth is not None:
                    ax.axvline(truth[j], color=INK, lw=0.5, alpha=0.5)
                    ax.axhline(truth[i], color=INK, lw=0.5, alpha=0.5)
                    ax.plot(truth[j], truth[i], "s", ms=3, color=INK)
                ax.set_ylim(lo[a], up[a])
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, prune="both"))
            ax.set_xlim(lo[b], up[b])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, prune="both"))
            if a == n - 1:
                ax.set_xlabel(axis_label(model, j, n <= 12), fontsize=fs)
                ax.tick_params(axis="x", labelsize=fs - 1.5, rotation=0 if n <= 8 else 45)
            else:
                ax.set_xticklabels([])
            if b == 0 and a > 0:
                ax.set_ylabel(axis_label(model, i, n <= 12), fontsize=fs)
                ax.tick_params(axis="y", labelsize=fs - 1.5)
            else:
                ax.set_yticklabels([])

    handles = [matplotlib.lines.Line2D([], [], color=s["color"], ls=s["ls"], lw=1.3, label=s["label"]) for s in sets]
    if truth is not None:
        handles.append(matplotlib.lines.Line2D([], [], color=INK, marker="s", ms=3, lw=0.8, label="Truth"))
    w, h = (right - left) / n, (top - bottom) / n
    x0 = left + w * (n / 2 + 0.4)
    if spectrum is not None and 4 <= n <= 12:  # the observation, in the empty upper triangle
        x_obs, preds = spectrum
        y0 = top - h * (n // 3) + 0.06
        ins = fig.add_axes([x0, y0, right - x0, top - y0 - 0.02])
        for s, f in zip(sets, preds):
            ins.plot(model.wavelength, np.median(f, 0), color=s["color"], ls=s["ls"], lw=1.0)
        ins.errorbar(model.wavelength, x_obs, model.sigma, fmt="o", ms=2, color=INK, elinewidth=0.5, zorder=5)
        ins.set_xlabel("Wavelength [μm]", fontsize=fs)
        ins.set_ylabel("Flux", fontsize=fs)
        ins.tick_params(labelsize=fs - 1.5)
        ins.set_title(f"Observation and median spectra ({model.name})", fontsize=fs + 1, loc="left")
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, y0 - 0.07), fontsize=fs + 1)
    else:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(right, top), fontsize=10)
        fig.suptitle(f"Posteriors on {model.name} (68% and 95% contours)", fontsize=12, x=left, ha="left")
    fig.savefig(path, dpi=80 if n > 15 else 110)
    plt.close(fig)
    print(path)


def ppc(model: Model, sets: list, preds: list, x_obs: np.ndarray, truth, path: str) -> list:
    """Spectra of posterior draws against x_obs, and the χ² of x_obs under each draw. Returns the χ²."""
    rng = np.random.default_rng(1)
    wl, sigma = model.wavelength, model.sigma
    fig, (ax, axr, axc) = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw=dict(height_ratios=[2.4, 1, 1.3],
                                                                                 hspace=0.5))
    chi2, p_values = [], []
    for s, f in zip(sets, preds):
        x_rep = f + sigma * rng.standard_normal(f.shape)  # replicated data: model + new noise
        chi_obs = np.sum(((x_obs - f) / sigma) ** 2, axis=1)
        chi2.append(chi_obs)
        p_values.append(np.mean(np.sum(((x_rep - f) / sigma) ** 2, axis=1) >= chi_obs))
        ax.plot(wl, np.median(f, 0), color=s["color"], ls=s["ls"], lw=1.2)
        axr.plot(wl, np.median((x_obs - f) / sigma, 0), color=s["color"], ls=s["ls"], lw=1.0)
    ax.errorbar(wl, x_obs, sigma, fmt="o", ms=3, color=INK, elinewidth=0.7, zorder=5)
    ax.set_title(f"Posterior predictive check ({model.name})", loc="left")
    ax.set_ylabel("Flux [10$^{-16}$ W m$^{-2}$ μm$^{-1}$]")
    handles = [matplotlib.lines.Line2D([], [], color=INK, marker="o", ms=3, ls="", label="Observation ± σ")]
    handles += [matplotlib.lines.Line2D([], [], color=s["color"], ls=s["ls"], label=f"{s['label']}, median")
                for s in sets]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    axr.axhspan(-2, 2, color=FAINT, lw=0)
    axr.axhspan(-1, 1, color="#dddcd7", lw=0)
    axr.axhline(0, color=MUTED, lw=0.6)
    axr.set_ylim(-3.5, 3.5)
    axr.set_ylabel("Residual / σ")
    axr.set_xlabel("Wavelength [μm]")
    axr.set_title("Median residual (bands: ±1σ, ±2σ)", loc="left", fontsize=9, color=MUTED)
    for a in (ax, axr):
        a.set_xlim(wl[0] - 0.02, wl[-1] + 0.02)

    every = np.concatenate(chi2)
    edges = np.linspace(every.min(), np.quantile(every, 0.98), 30)
    for s, c, p in zip(sets, chi2, p_values):
        axc.hist(c, edges, density=True, color=s["color"], ls=s["ls"], histtype="step", lw=1.2,
                 label=f"{s['label']}, median {np.median(c):.0f}, p = {p:.2f}")
    marks = [(model.n_bins, "Number of bins")]
    if truth is not None:
        marks.append((np.sum(((x_obs - model.flux(truth)[0]) / sigma) ** 2), "At the truth"))
    for x, text in marks:
        if edges[0] <= x <= edges[-1]:
            axc.axvline(x, color=INK, lw=0.8, ls=":" if text == "Number of bins" else "-")
            axc.text(x, 1.0, f" {text} ({x:.0f})", transform=axc.get_xaxis_transform(), fontsize=8, va="top")
    axc.set_yticks([])
    axc.spines["left"].set_visible(False)
    axc.set_xlabel("χ² of the observation against posterior draws")
    axc.set_title("Fit (p: posterior predictive p-value)", loc="left", fontsize=9, color=MUTED)
    axc.legend(fontsize=7.5, loc="upper right")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return chi2


def table(sets: list, chi2: list, path: str, against: str | None):
    """Cost, importance-sampling ESS, log Z, fit, and the marginals compared with those of the run `against`
    (a comparison between methods, not with the truth): mean shift = mean over parameters of |Δ mean| / σ_against,
    width = median over parameters of σ / σ_against."""
    ref = next((s for s in sets if s["name"] == against and not s["q"]), None)
    name = LABELS.get(against, against)
    name = name[0].lower() + name[1:] if name[1:2].islower() else name  # "nested sampling", "NPE"
    rows = ["| Posterior | Simulations | ESS | log Z | Median χ² |"
            + (f" Mean shift vs {name} (σ) | Width vs {name} |" if ref else ""),
            "|---|---|---|---|---|" + ("---|---|" if ref else "")]
    for s, c in zip(sets, chi2):
        su = s["summary"]
        ess = f"{su['ess']:,.0f}" if "ess" in su else "–"
        logz = f"{su['logz']:.2f}" + (f" ± {su['logz_err']:.2f}" if "logz_err" in su else "")
        if s["q"]:
            ess = logz = "–"  # q itself: the ESS and log Z belong to its reweighted version
        row = f"| {s['label']} | {su['n_sims']:,} | {ess} | {logz} | {np.median(c):.0f} |"
        if ref:
            dm = ds = "–"
            if s is not ref:
                sd = ref["samples"].std(0)
                dm = f"{np.mean(np.abs(s['samples'].mean(0) - ref['samples'].mean(0)) / sd):.2f}"
                ds = f"{np.median(s['samples'].std(0) / sd):.2f}"
            row += f" {dm} | {ds} |"
        rows.append(row)
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def main():
    p = runs.parser(__doc__)
    p.add_argument("--runs", nargs="+", help="run names in runs/<model>/ (default: all, in alphabetical order)")
    p.add_argument("--against", default="nested", help="the run the table compares the marginals with "
                                                       "(default: nested)")
    p.add_argument("--only-is", action="store_true", help="NPE / TSNPE: draw the reweighted samples only")
    p.add_argument("--params", nargs="+", help="parameters of the corner plot (default: all if at most 12, "
                                               "else the 12 the first posterior constrains most)")
    p.add_argument("--out", default="runs", help="the runs are in OUT/<model>/ (default: runs)")
    args = p.parse_args()
    model = Model(args.model)
    folder = os.path.join(args.out, args.model)
    names = args.runs or sorted(r for r in os.listdir(folder) if os.path.exists(f"{folder}/{r}/summary.json"))
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
    rng = np.random.default_rng(0)  # predicted spectra: 300 draws of each posterior
    preds = [model.flux(s["samples"][rng.choice(len(s["samples"]), min(300, len(s["samples"])), replace=False)])
             for s in sets]
    with plt.rc_context(STYLE):
        corner(model, sets, idx, truth, f"{folder}/corner.png", spectrum=(x_obs, preds))
        if model.D > 12 and not args.params:
            corner(model, sets, list(range(model.D)), truth, f"{folder}/corner-all.png")
        chi2 = ppc(model, sets, preds, x_obs, truth, f"{folder}/ppc.png")
    table(sets, chi2, f"{folder}/table.md", args.against)


if __name__ == "__main__":
    main()
