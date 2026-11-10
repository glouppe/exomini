# Parameters and priors

Generated from the `PARAMETERS` tables in `exomini/exo*.py` by `python tools/make_parameters_md.py`.
All priors are uniform. θ* is the benchmark parameter vector (test pair 0).

## exo6

A 6-parameter emission spectrum of a warm giant planet, with self-consistent physics.

94 bins over 0.95–2.42 μm, noise σ = 0.250 × 1e-16 W m⁻² μm⁻¹ per bin.

| | parameter | prior | θ* | unit | description |
|---|---|---|---|---|---|
| 0 | `T_int` | U(700, 1700) | 1100 | K | internal temperature, sets the P–T profile |
| 1 | `log_g` | U(3, 5) | 3.75 | log10 cm/s² | surface gravity |
| 2 | `MH` | U(-1, 1) | 0 | dex | metallicity [M/H]: C and O scale with 10^[M/H] |
| 3 | `CO` | U(0.2, 1.2) | 0.55 |  | carbon-to-oxygen ratio |
| 4 | `log_Pc` | U(-1.5, 1.5) | 0.5 | log10 bar | pressure at the top of an opaque grey cloud deck |
| 5 | `R_p` | U(0.8, 1.6) | 1 | R_Jup | planet radius |

## exo30

A 30-parameter emission spectrum, shaped like a free retrieval.

176 bins over 0.85–4.92 μm, noise σ = 0.392 × 1e-16 W m⁻² μm⁻¹ per bin.

| | parameter | prior | θ* | unit | description |
|---|---|---|---|---|---|
| 0 | `log_g` | U(3, 5) | 3.75 | log10 cm/s² | surface gravity |
| 1 | `R_p` | U(0.8, 1.6) | 1 | R_Jup | planet radius |
| 2 | `T_0` | U(1500, 6000) | 4900 | K | temperature at the bottom node, 100 bar |
| 3 | `r_1` | U(0.55, 1.05) | 0.62 |  | T_1 / T_0, temperature ratio between P–T nodes 0 and 1 |
| 4 | `r_2` | U(0.55, 1.05) | 0.66 |  | T_2 / T_1, temperature ratio between P–T nodes 1 and 2 |
| 5 | `r_3` | U(0.55, 1.05) | 0.72 |  | T_3 / T_2, temperature ratio between P–T nodes 2 and 3 |
| 6 | `r_4` | U(0.55, 1.05) | 0.8 |  | T_4 / T_3, temperature ratio between P–T nodes 3 and 4 |
| 7 | `r_5` | U(0.55, 1.05) | 0.88 |  | T_5 / T_4, temperature ratio between P–T nodes 4 and 5 |
| 8 | `r_6` | U(0.55, 1.05) | 0.95 |  | T_6 / T_5, temperature ratio between P–T nodes 5 and 6 |
| 9 | `r_7` | U(0.55, 1.05) | 0.98 |  | T_7 / T_6, temperature ratio between P–T nodes 6 and 7 |
| 10 | `r_8` | U(0.55, 1.05) | 1 |  | T_8 / T_7, temperature ratio between P–T nodes 7 and 8 |
| 11 | `r_9` | U(0.55, 1.05) | 1 |  | T_9 / T_8, temperature ratio between P–T nodes 8 and 9 |
| 12 | `log_H2O` | U(-10, -2) | -3.3 | log10 VMR | H2O volume mixing ratio, constant with altitude |
| 13 | `log_CH4` | U(-10, -2) | -3.6 | log10 VMR | CH4 volume mixing ratio, constant with altitude |
| 14 | `log_CO` | U(-10, -2) | -3.5 | log10 VMR | CO volume mixing ratio, constant with altitude |
| 15 | `log_CO2` | U(-10, -2) | -5 | log10 VMR | CO2 volume mixing ratio, constant with altitude |
| 16 | `log_NH3` | U(-10, -2) | -4.5 | log10 VMR | NH3 volume mixing ratio, constant with altitude |
| 17 | `log_H2S` | U(-10, -2) | -4.5 | log10 VMR | H2S volume mixing ratio, constant with altitude |
| 18 | `log_HCN` | U(-10, -2) | -5 | log10 VMR | HCN volume mixing ratio, constant with altitude |
| 19 | `log_C2H2` | U(-10, -2) | -6 | log10 VMR | C2H2 volume mixing ratio, constant with altitude |
| 20 | `log_PH3` | U(-10, -2) | -5.5 | log10 VMR | PH3 volume mixing ratio, constant with altitude |
| 21 | `log_FeH` | U(-10, -2) | -6 | log10 VMR | FeH volume mixing ratio, constant with altitude |
| 22 | `log_CrH` | U(-10, -2) | -6.5 | log10 VMR | CrH volume mixing ratio, constant with altitude |
| 23 | `log_TiO` | U(-10, -2) | -8 | log10 VMR | TiO volume mixing ratio, constant with altitude |
| 24 | `log_VO` | U(-10, -2) | -8 | log10 VMR | VO volume mixing ratio, constant with altitude |
| 25 | `log_K` | U(-10, -2) | -5.5 | log10 VMR | K volume mixing ratio, constant with altitude |
| 26 | `log_SiO` | U(-10, -2) | -5 | log10 VMR | SiO volume mixing ratio, constant with altitude |
| 27 | `log_Pc` | U(-2, 2) | 0.5 | log10 bar | pressure at the top of the cloud deck |
| 28 | `log_tau_c` | U(-1, 2) | 0.5 | log10 | cloud optical depth per dex of pressure, at 1 μm |
| 29 | `alpha_c` | U(0, 4) | 1 |  | cloud slope: τ ∝ (λ / 1 μm)^-alpha_c |

## exo50

A 50-parameter emission spectrum of a brown dwarf, in the style of petitRADTRANS retrievals.

176 bins over 0.85–4.92 μm, noise σ = 0.300 × 1e-16 W m⁻² μm⁻¹ per bin.

| | parameter | prior | θ* | unit | description |
|---|---|---|---|---|---|
| 0 | `R_p` | U(0.8, 1.6) | 1 | R_Jup | planet radius |
| 1 | `log_g` | U(3, 5.5) | 4.5 | log10 cm/s² | surface gravity |
| 2 | `T_bottom` | U(2000, 8000) | 4500 | K | temperature at 100 bar |
| 3 | `PTslope_1` | U(0, 0.45) | 0.3 |  | d ln T / d ln P at node 1 (node 1: 100 bar, node 10: 1e-5 bar) |
| 4 | `PTslope_2` | U(0, 0.45) | 0.28 |  | d ln T / d ln P at node 2 (node 1: 100 bar, node 10: 1e-5 bar) |
| 5 | `PTslope_3` | U(0, 0.45) | 0.25 |  | d ln T / d ln P at node 3 (node 1: 100 bar, node 10: 1e-5 bar) |
| 6 | `PTslope_4` | U(0, 0.45) | 0.2 |  | d ln T / d ln P at node 4 (node 1: 100 bar, node 10: 1e-5 bar) |
| 7 | `PTslope_5` | U(0, 0.45) | 0.15 |  | d ln T / d ln P at node 5 (node 1: 100 bar, node 10: 1e-5 bar) |
| 8 | `PTslope_6` | U(0, 0.45) | 0.12 |  | d ln T / d ln P at node 6 (node 1: 100 bar, node 10: 1e-5 bar) |
| 9 | `PTslope_7` | U(0, 0.45) | 0.1 |  | d ln T / d ln P at node 7 (node 1: 100 bar, node 10: 1e-5 bar) |
| 10 | `PTslope_8` | U(0, 0.45) | 0.08 |  | d ln T / d ln P at node 8 (node 1: 100 bar, node 10: 1e-5 bar) |
| 11 | `PTslope_9` | U(0, 0.45) | 0.06 |  | d ln T / d ln P at node 9 (node 1: 100 bar, node 10: 1e-5 bar) |
| 12 | `PTslope_10` | U(0, 0.45) | 0.05 |  | d ln T / d ln P at node 10 (node 1: 100 bar, node 10: 1e-5 bar) |
| 13 | `MH` | U(-1, 1) | 0 | dex | metallicity [M/H]: C and O scale with 10^[M/H] |
| 14 | `CO` | U(0.2, 1.2) | 0.55 |  | carbon-to-oxygen ratio |
| 15 | `log_Pquench` | U(-3, 2) | 0.5 | log10 bar | quench pressure: H2O, CH4, CO frozen above it |
| 16 | `log_12C_13C` | U(1, 2.5) | 1.95 | log10 | isotope ratio 12C / 13C, sets 13CO from CO |
| 17 | `log_CO2` | U(-12, -2) | -5.5 | log10 VMR | CO2 volume mixing ratio, constant with altitude |
| 18 | `log_NH3` | U(-12, -2) | -5 | log10 VMR | NH3 volume mixing ratio, constant with altitude |
| 19 | `log_H2S` | U(-12, -2) | -4.5 | log10 VMR | H2S volume mixing ratio, constant with altitude |
| 20 | `log_HCN` | U(-12, -2) | -6 | log10 VMR | HCN volume mixing ratio, constant with altitude |
| 21 | `log_C2H2` | U(-12, -2) | -8 | log10 VMR | C2H2 volume mixing ratio, constant with altitude |
| 22 | `log_PH3` | U(-12, -2) | -6.5 | log10 VMR | PH3 volume mixing ratio, constant with altitude |
| 23 | `log_SiO` | U(-12, -2) | -6 | log10 VMR | SiO volume mixing ratio, constant with altitude |
| 24 | `log_FeH` | U(-12, -2) | -7 | log10 VMR | FeH volume mixing ratio, constant with altitude |
| 25 | `log_CrH` | U(-12, -2) | -8.5 | log10 VMR | CrH volume mixing ratio, constant with altitude |
| 26 | `log_TiO` | U(-12, -2) | -10 | log10 VMR | TiO volume mixing ratio, constant with altitude |
| 27 | `log_VO` | U(-12, -2) | -10 | log10 VMR | VO volume mixing ratio, constant with altitude |
| 28 | `log_K` | U(-12, -2) | -6 | log10 VMR | K volume mixing ratio, constant with altitude |
| 29 | `log_Na` | U(-12, -2) | -5.5 | log10 VMR | Na volume mixing ratio, constant with altitude |
| 30 | `log_Cs` | U(-12, -2) | -9 | log10 VMR | Cs volume mixing ratio, constant with altitude |
| 31 | `log_OH` | U(-12, -2) | -6.5 | log10 VMR | OH volume mixing ratio, constant with altitude |
| 32 | `log_HF` | U(-12, -2) | -7 | log10 VMR | HF volume mixing ratio, constant with altitude |
| 33 | `log_HCl` | U(-12, -2) | -7.5 | log10 VMR | HCl volume mixing ratio, constant with altitude |
| 34 | `log_AlO` | U(-12, -2) | -10 | log10 VMR | AlO volume mixing ratio, constant with altitude |
| 35 | `log_CaH` | U(-12, -2) | -9.5 | log10 VMR | CaH volume mixing ratio, constant with altitude |
| 36 | `log_OCS` | U(-12, -2) | -8 | log10 VMR | OCS volume mixing ratio, constant with altitude |
| 37 | `log_X_cb_MgSiO3` | U(-6, -1) | -3 | log10 | MgSiO3(s) mass fraction at the cloud base |
| 38 | `log_Pbase_MgSiO3` | U(-1, 2) | 1.3 | log10 bar | MgSiO3(s) cloud-base pressure |
| 39 | `f_sed_MgSiO3` | U(0.5, 8) | 2 |  | MgSiO3(s) sedimentation efficiency: X ∝ (P / P_base)^f_sed |
| 40 | `log_cloud_radius_MgSiO3` | U(-2, 1) | 0 | log10 μm | MgSiO3(s) median particle radius |
| 41 | `log_X_cb_Fe` | U(-6, -1) | -3.5 | log10 | Fe(s) mass fraction at the cloud base |
| 42 | `log_Pbase_Fe` | U(-1, 2) | 1.6 | log10 bar | Fe(s) cloud-base pressure |
| 43 | `f_sed_Fe` | U(0.5, 8) | 2 |  | Fe(s) sedimentation efficiency: X ∝ (P / P_base)^f_sed |
| 44 | `log_cloud_radius_Fe` | U(-2, 1) | -0.3 | log10 μm | Fe(s) median particle radius |
| 45 | `sigma_lnorm` | U(1.05, 3) | 2 |  | width of the log-normal particle size distribution, both clouds |
| 46 | `f_cloud` | U(0, 1) | 0.7 |  | fraction of the disc covered by clouds |
| 47 | `log_tau_haze` | U(-4, 0) | -2.5 | log10 | haze optical depth per dex of pressure, at 1 μm |
| 48 | `alpha_haze` | U(0, 6) | 4 |  | haze slope: τ ∝ (λ / 1 μm)^-alpha_haze |
| 49 | `s_red` | U(0.9, 1.1) | 1.02 |  | flux scale of the red detector (λ > 2.9 μm) relative to the blue one |
