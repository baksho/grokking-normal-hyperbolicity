"""Plot accuracy and the normal restoring curvature against training step.

Accepts one or more results CSVs (e.g. several seeds). With multiple seeds, the
test-accuracy and sigma_min^+ curves are overlaid so a per-seed dip (or its
absence) is visible.
"""
import argparse, csv, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    steps, tra, tea, smin = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            tra.append(float(row["train_acc"]))
            tea.append(float(row["test_acc"]))
            smin.append(float(row["sigma_min_plus"]))
    return steps, tra, tea, smin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results/results_seed0.csv"],
                    help="CSV path(s); globs allowed")
    ap.add_argument("--band", type=float, nargs=2, default=[6600, 17400],
                    help="approx step range of the transition to shade")
    ap.add_argument("--fig", type=str, default="paper/figures/grokking_normal_hyperbolicity.png")
    args = ap.parse_args()

    paths = []
    for p in args.results:
        paths += sorted(glob.glob(p)) or [p]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(args.band[0], args.band[1], color="gold", alpha=0.18, label="grokking transition")
    ax2 = ax.twinx()
    for k, path in enumerate(paths):
        steps, tra, tea, smin = load(path)
        lbl = "" if len(paths) == 1 else f" (seed {k})"
        if k == 0:
            ax.plot(steps, tra, color="#888", lw=1.1, ls="--", label="train acc")
        ax.plot(steps, tea, color="#1f77b4", lw=2, alpha=0.9, label=f"test acc{lbl}")
        ax2.plot(steps, smin, color="#d62728", lw=2, marker="o", ms=2.5, alpha=0.9,
                 label=fr"$\sigma_{{\min}}^+(J)${lbl}")

    ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.03, 1.03)
    ax2.set_ylabel(r"$\sigma_{\min}^+(J)$  (normal restoring curvature)", color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8, framealpha=0.9)
    ax.set_title("Grokking transition vs. manifold normal hyperbolicity")
    plt.tight_layout()
    plt.savefig(args.fig, dpi=140)
    print(f"saved {args.fig}")


if __name__ == "__main__":
    main()
