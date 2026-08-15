#!/usr/bin/env bash
# End-to-end reproduction of both paper figures.
set -e
pip install -r requirements.txt
python src/train.py --seed 0 --out runs                       # train + snapshot
python src/diagnostic.py --run runs/run_seed0.pt --out results # sigma_1..6 per snapshot
python src/multiseed.py --seeds 0 1 2 3 4 --out results        # 5-seed robustness
python src/plot.py --results results/results_seed0.csv \
    --multiseed results/multiseed.csv --figdir paper/figures
echo "Figures written to paper/figures/"
