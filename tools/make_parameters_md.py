"""Writes PARAMETERS.md from the PARAMETERS tables of the models. Run from the repository root:

    python tools/make_parameters_md.py
"""

from exomini import MODELS, Model

out = ["# Parameters and priors", "",
       "Generated from the `PARAMETERS` tables in `exomini/exo*.py` by `python tools/make_parameters_md.py`.",
       "All priors are uniform. θ* is the benchmark parameter vector (test pair 0).", ""]
for name in MODELS:
    m = Model(name, device="cpu")
    title = m.module.__doc__.strip().split("\n")[0].split(": ", 1)[1]
    out += [f"## {name}", "", title[0].upper() + title[1:], "",
            f"{m.n_bins} bins over {m.wavelength[0]:.2f}–{m.wavelength[-1]:.2f} μm, "
            f"noise σ = {m.sigma:.3f} × 1e-16 W m⁻² μm⁻¹ per bin.", "", m.describe(), ""]
with open("PARAMETERS.md", "w") as f:
    f.write("\n".join(out))
print("PARAMETERS.md")
