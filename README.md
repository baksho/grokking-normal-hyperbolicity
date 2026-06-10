# Is Grokking a Loss of Normal Hyperbolicity of the Interpolation Manifold?

Code and figure for the SKILL 2026 short paper that tests whether the sharp
generalization transition in grokking coincides with a loss of normal
hyperbolicity of the zero-loss (interpolation) manifold.

## About

The post-memorization phase of grokking is often modelled as a fast--slow
system: a fast process pulls the parameters onto the interpolation manifold,
and a slow weight-decay-driven drift moves them along it. This repository asks
whether the sudden transition is a *fold/bifurcation* of that slow manifold
(its normal restoring curvature collapsing) or smooth drift across a
persistently attracting manifold. It measures one scalar along training,
`sigma_min^+(J)` — the smallest nonzero singular value of the residual Jacobian,
which equals the smallest normal restoring curvature — on a two-layer ReLU
network that groks modular addition under squared loss. In the regime studied,
`sigma_min^+(J)` does not collapse at the transition; it is small only before
memorization and largest while the model generalizes.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.2+ and Matplotlib (see `requirements.txt`). CPU is sufficient; the
  default run is small.

### Installation

```bash
git clone https://github.com/your-username/grokking-normal-hyperbolicity.git
cd grokking-normal-hyperbolicity
pip install -r requirements.txt
```

### Reproduce the figure

```bash
bash run.sh
```

This trains the default configuration, computes the diagnostic on the
snapshots, and writes `paper/figures/grokking_normal_hyperbolicity.png`.

## Usage

The three stages can also be run separately:

```bash
# 1. Train and snapshot parameters across training
python src/train.py --seed 0 --out runs

# 2. Compute sigma_min^+(J) on each snapshot -> results/results_seed0.csv
python src/diagnostic.py --run runs/run_seed0.pt --out results

# 3. Plot accuracy and sigma_min^+(J) vs. step
python src/plot.py --results results/results_seed0.csv \
    --fig paper/figures/grokking_normal_hyperbolicity.png
```

### Multiple seeds

The paper figure is single-seed; a robust claim needs several. Train a few
seeds and overlay their curves:

```bash
for s in 0 1 2 3 4; do
  python src/train.py --seed $s --out runs
  python src/diagnostic.py --run runs/run_seed$s.pt --out results
done
python src/plot.py --results "results/results_seed*.csv" --fig multi_seed.png
```

If `sigma_min^+(J)` stays bounded away from zero across all seeds, the negative
result holds; if any seed dips at its transition, that is a different and more
interesting finding.

### Key arguments (`src/train.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--p` | 11 | modulus for addition mod p |
| `--width` | 96 | hidden width |
| `--train_frac` | 0.7 | fraction of the p^2 pairs used for training |
| `--init_scale` | 3.5 | multiplier on Kaiming init (large init lengthens the plateau) |
| `--weight_decay` | 2.0 | AdamW decoupled weight decay |
| `--lr` | 3e-3 | AdamW learning rate |
| `--steps` | 35000 | training steps |
| `--snapshot_every` | 600 | snapshot interval (also the diagnostic resolution) |

## Project Structure

```
.
├── run.sh                 # end-to-end reproduction
├── requirements.txt
├── src/
│   ├── train.py           # train + snapshot theta
│   ├── diagnostic.py      # sigma_min^+(J) per snapshot -> CSV
│   └── plot.py            # accuracy + sigma_min^+(J) figure
└── paper/
    ├── grokking_normal_hyperbolicity_skill2026.tex
    └── figures/grokking_normal_hyperbolicity.png
```

## Results

On the default run the network memorizes by ~step 4k (train accuracy 1.0, test
accuracy 0.0 through ~6k), then test accuracy climbs to ~0.95 over steps
~7k–17k. Across that transition `sigma_min^+(J)` sits at its maximum
(~0.20–0.23); it is near zero (~0.007) only before memorization. The
interpolation manifold therefore stays normally hyperbolic through grokking in
this regime — evidence against the bifurcation hypothesis, though single-seed
and under AdamW rather than the gradient descent of the underlying theory. See
the paper's limitations section for the bounds on this claim.

## Paper

The LaTeX source is in `paper/`. It compiles as-is with `pdflatex` (article
class) for preview; for submission, port the body into the official GI-LNI
template. Verify all bibliography entries before submitting, in particular the
2026 preprints.

## License

MIT — see [LICENSE](LICENSE).
