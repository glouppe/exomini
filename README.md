# toysbi

Simulation-based inference on toy exoplanet emission spectra. It provides three forward models (6, 30 and 50
parameters) with exact likelihoods, and three inference methods:
- **NPE**: amortised neural posterior estimation.
- **TSNPE**: truncated sequential NPE.
- **Nested sampling**: the likelihood-based reference.

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

| | `exo6` | `exo30` | `exo50` |
|---|---|---|---|
| Temperature | grey, from T_int | 10 nodes | 10 nodes (gradients) |
| Chemistry | [M/H], C/O | 15 free species | [M/H], C/O, quench, 13CO, 20 free species |
| Clouds | grey deck | deck | MgSiO3 + Fe, patchy; haze |
| Wavelengths | 0.95–2.45 μm, 94 bins | 0.85–4.95 μm, 176 bins | as `exo30` |

[PARAMETERS.md](PARAMETERS.md) lists every parameter: its prior, unit, meaning and benchmark value.

Priors are uniform. To change one, edit the `PARAMETERS` table at the top of `toysbi/exo*.py`:

```python
# name, prior lower, prior upper, benchmark θ*, unit, description
("log_g", 3.0, 5.0, 3.75, "log10 cm/s²", "surface gravity"),
```

The noise is Gaussian with a fixed σ per bin, giving S/N ≈ 30 at the benchmark's peak.

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

## Findings on exo30, at 1M simulations

```bash
python npe.py   --model exo30 --sims 1000000
python tsnpe.py --model exo30 --rounds 5 --sims 200000
python nested.py --model exo30 --nlive 2000
```

- **NPE fails**: ESS 1–6 of 50,000, posteriors too broad, χ² ≈ 215 for 176 bins.
- **TSNPE works**: ESS 2,000–3,600, log Z within 0.1 over four runs, usable from 400k simulations.
- **Nested sampling is not ground truth**: runs with 1,000 and 2,000 live points differ by 1.0 in log Z.

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
