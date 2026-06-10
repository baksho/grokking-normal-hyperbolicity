#!/usr/bin/env bash
# End-to-end reproduction of the paper figure (single seed).
set -e
pip install -r requirements.txt
python src/train.py --seed 0 --out runs
python src/diagnostic.py --run runs/run_seed0.pt --out results
python src/plot.py --results results/results_seed0.csv --fig paper/figures/grokking_normal_hyperbolicity.png
echo "Figure written to paper/figures/grokking_normal_hyperbolicity.png"
