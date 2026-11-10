# toysbi

Simulation-based inference on toy exoplanet emission spectra. It provides three forward models (6, 30 and 50
parameters) with exact likelihoods, and three inference methods:
- **NPE**: amortised neural posterior estimation.
- **TSNPE**: truncated sequential NPE.
- **Nested sampling**: the likelihood-based reference.

> **Disclaimer.** This codebase was entirely vibe-coded with an AI assistant. The physics is schematic and the code is not
> validated for real data: do not use it for serious exoplanet science. Its purpose is to demonstrate how to do
> simulation-based inference (NPE, TSNPE) and nested sampling on a problem shaped like an atmospheric retrieval.

## Install

```bash
pip install -e ".[test]"
pytest
```

The code runs on CPU and uses a GPU when there is one.

## Quick start

```bash
python npe.py     --model exo6   # amortised NPE
python tsnpe.py   --model exo6   # truncated sequential NPE
python nested.py  --model exo6   # nested sampling
python compare.py --model exo6   # corner.png, ppc.png, table.md
```

Results go to `runs/exo6/`.

## Models and priors

A model maps parameters θ to the spectrum a telescope would see:

    θ  →  atmosphere (temperature at each altitude, gas abundances, clouds)  →  emitted light  →  binned, plus noise

The three models are the same kind of planet at increasing levels of detail.

**`exo6`** has 6 parameters and a few physical assumptions. One internal temperature sets the whole
temperature profile. The overall metallicity and the carbon-to-oxygen ratio set the abundances of H2O, CH4 and
CO. The remaining parameters are a cloud-top pressure, the surface gravity and the radius. The parameters are
well constrained, so this model is quick to explore.

**`exo30`** has 30 parameters and few assumptions, like a "free" retrieval:
- Temperature: free at 10 altitudes.
- Gases: 15 species, each abundance free.
- Cloud: one deck with free height, thickness and colour.
- Planet: gravity and radius.

Many parameters are barely constrained by the data, as in practice.

**`exo50`** has 50 parameters, modelled on retrievals of brown dwarfs with petitRADTRANS:
- Temperature: its slope with altitude, free at 10 altitudes.
- Main gases: from metallicity and C/O, with mixing frozen above a free pressure, plus the 13C isotope.
- Trace gases: 20 species, each abundance free.
- Clouds: silicate and iron, each with a free base, amount, thickness and particle size. They cover a free
  fraction of the disc, and there is a haze.
- Instrument: a calibration offset between the two detectors.

About half the parameters are constrained.

| | `exo6` | `exo30` | `exo50` |
|---|---|---|---|
| Parameters | 6 | 30 | 50 |
| Wavelengths | 0.95–2.45 μm | 0.85–4.95 μm | 0.85–4.95 μm |
| Spectral bins (R ≈ 100) | 94 | 176 | 176 |

**Every parameter, with its meaning, unit, prior range and benchmark value θ\*, is listed in
[PARAMETERS.md](PARAMETERS.md).** For a single model, run `python simulate.py --model exo50 --describe`.

**Priors** are uniform between a lower and an upper bound. They are set in the `PARAMETERS` table at the top of
`toysbi/exo6.py`, `exo30.py` and `exo50.py`, one row per parameter. To change a prior, edit its bounds:

```python
# name, prior lower, prior upper, benchmark θ*, unit, description
("log_g", 3.0, 5.0, 3.75, "log10 cm/s²", "surface gravity"),
```

**Noise** is Gaussian, with the same σ in every bin: S/N ≈ 30 at the brightest point of the benchmark spectrum.

## Data

```python
from toysbi import Model

m = Model("exo30")
theta = m.sample_prior(1000, seed=0)
x = m.simulate(theta, seed=0)    # noisy spectra; m.flux(theta) gives them noise-free
theta, x = m.test_pairs()        # 8 fixed pairs; pair 0 is the benchmark
m.log_likelihood(theta, x)       # exact
```

```bash
python simulate.py --model exo50 --describe                  # parameter table
python simulate.py --model exo6 --n 10000 --out data.npz     # pairs from the prior
python simulate.py --model exo6 --theta 1300 4 0.5 0.8 0 1.2 --out obs.npz
```

By default every method conditions on test pair 0. Choose another with `--obs 3`, or your own with
`--obs obs.npz`.

## Methods

- **`npe.py`**: trains q(θ | x) once on prior simulations, then samples q(θ | x_obs).
- **`tsnpe.py`**: works in rounds. Each round simulates from the prior restricted to where the last
  q(θ | x_obs) puts 1 − 10⁻⁴ of its mass, then retrains on all the simulations so far.
- **`nested.py`**: dynesty with the exact likelihood. It gives log Z, at one simulation per likelihood
  evaluation.

NPE and TSNPE finish with **importance sampling**: q's draws are reweighted by p(x | θ) p(θ) / q(θ | x).
- **ESS** close to the number of draws means q is accurate; an ESS of a few means q is wrong.
- **log Z** is the evidence.

The options shared by all scripts are `--model`, `--obs`, `--seed`, `--out` and `--name`. See `--help`.

## Results

`compare.py` writes three files to `runs/<model>/`:
- **`corner.png`**: marginals. Nested sampling is grey; for NPE and TSNPE, q is solid and q + IS is dashed.
- **`ppc.png`**: spectra of posterior draws, and the χ² of x_obs. A good posterior gives χ² ≈ the number of bins.
- **`table.md`**: simulations used, ESS, log Z, χ², and the distance of the marginals to nested sampling.

## Layout

| | |
|---|---|
| `toysbi/exo6.py`, `exo30.py`, `exo50.py` | models: `PARAMETERS`, grids, batched `flux(θ)` in torch |
| `toysbi/physics.py` | radiative transfer, opacity bands, binning |
| `toysbi/model.py` | `Model`: prior, simulation, likelihood, test pairs |
| `toysbi/inference.py` | `NPE`, `fit`, `importance_sampling`, `truncated_prior` |
| `toysbi/runs.py` | run folders, shared script options |
| `toysbi/data/` | the test pairs (`tests/make_test_pairs.py`) |
| `*.py` | the scripts |
| `tests/` | models, inference, scripts end to end |
