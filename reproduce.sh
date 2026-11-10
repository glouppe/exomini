#!/bin/bash
# Reproduces results/exo6/: each method on exo6 with its default settings (seed 0), then the comparison.
# The three methods are independent: on a cluster, run them as parallel jobs.
set -euo pipefail
python npe.py     --model exo6
python tsnpe.py   --model exo6
python nested.py  --model exo6
python compare.py --model exo6
mkdir -p results/exo6
python simulate.py --model exo6 --plot results/exo6/parameters.png
cp runs/exo6/corner.png runs/exo6/ppc.png runs/exo6/table.md results/exo6/
